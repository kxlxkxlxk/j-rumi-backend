"""
ColorChecker Classic (24-patch) detection + color-correction calibration.

Pipeline:
  1. Propose several candidate quadrilaterals for where the card might be,
     using multiple independent segmentation strategies (see
     `_candidate_quads` below) -- deliberately liberal, since a real photo
     can have hair/clothing/jewelry that looks card-like to any one of
     them on its own.
  2. For EVERY candidate: perspective-warp it, detect the 24 patches
     inside, and check how well their colors match the card's actual
     known reference colors (Hungarian-matched CIEDE... well, Euclidean
     RGB distance here, then a proper affine solve). This is the real
     filter: a hair/clothing/jewelry region will not have anything
     resembling a match to the specific 24 reference colors, so its
     error stays high, while the true card (even from an imperfectly
     segmented candidate) matches closely.
  3. Keep only the candidate with the lowest resulting color-matching
     error. This makes automatic detection robust without needing a
     person to manually crop the card -- the known, fixed set of colors
     on the card is itself the strongest signal for "this is the card".

The same `calibrate_from_image()` function is used both when building the
foundation DB (photo of card + foundation swatch) and when a user submits
a face photo (photo of card + face) -- in both cases we first figure out
"what this camera/lighting did to a known color" and correct the *other*
thing in the same photo by the same transform.
"""

from dataclasses import dataclass
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from .reference_colors import (
    REFERENCE_RGB_LIST,
    REFERENCE_PATCHES,
    REFERENCE_NAMES,
    get_active_reference_rgb_list,
)

CANONICAL_W, CANONICAL_H = 1200, 800  # landscape canonical warp size (~3:2 card)

# The 6 most saturated ColorChecker Classic patches (the "primary/secondary"
# row: blue, green, red, yellow, magenta, cyan). Real skin, hair, and most
# clothing fall well outside these hues at this saturation, which makes
# them a strong, specific signal for "this pixel belongs to a card patch" --
# used by `_hue_grid_candidates` below to find the card by its own known
# colors directly, rather than by any generic brightness/texture heuristic.
_SATURATED_PATCH_NAMES = ("blue", "green", "red", "yellow", "magenta", "cyan")


def _target_hues():
    hues = []
    for name in _SATURATED_PATCH_NAMES:
        rgb = REFERENCE_PATCHES[name]
        bgr_px = np.uint8([[list(rgb[::-1])]])
        hsv_px = cv2.cvtColor(bgr_px, cv2.COLOR_BGR2HSV)[0, 0]
        hues.append(int(hsv_px[0]))
    return hues


_TARGET_HUES = _target_hues()


@dataclass
class CalibrationResult:
    success: bool
    message: str = ""
    correction_matrix: np.ndarray = None  # 3x4 affine (maps [R,G,B,1] observed -> corrected RGB)
    mean_delta_e: float = None
    card_corners: np.ndarray = None  # 4x2 in original image coords
    patch_centers_canonical: np.ndarray = None
    observed_patch_rgb: np.ndarray = None
    matched_reference_rgb: np.ndarray = None
    n_candidates_tried: int = None


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_candidates_from_mask(mask: np.ndarray, img_area: int, fill_thresh: float = 0.55, max_candidates: int = 6):
    """Return up to `max_candidates` plausible card-shaped quads from a
    binary mask, sorted largest-first. Deliberately loose (fill_thresh
    lower than you'd want for a final answer) -- real detection accuracy
    is enforced later by the color-matching check, not by geometry alone.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.005 or area > img_area * 0.95:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        quad = _order_quad_points(np.array(box))
        side_top = np.linalg.norm(quad[1] - quad[0])
        side_bottom = np.linalg.norm(quad[2] - quad[3])
        side_left = np.linalg.norm(quad[3] - quad[0])
        side_right = np.linalg.norm(quad[2] - quad[1])
        long_side = max(side_top, side_bottom, side_left, side_right)
        short_side = min(side_top, side_bottom, side_left, side_right)
        if short_side < 1:
            continue
        aspect = long_side / short_side
        # ColorChecker Classic is roughly 1.4-1.6 : 1 -- kept a bit wider
        # than that here since the color check downstream is the real filter.
        if 1.05 < aspect < 2.3:
            rect_area = short_side * long_side
            fill = area / rect_area if rect_area > 0 else 0
            if fill > fill_thresh:
                candidates.append((area, quad))
    candidates.sort(key=lambda x: -x[0])
    return [q for _, q in candidates[:max_candidates]]


def _color_variance_map(bgr_img: np.ndarray, win: int = 15) -> np.ndarray:
    """Local per-pixel color variance (sum over B,G,R of a windowed
    variance). The card's checkerboard is far more locally varied --
    tiny adjacent cells of wildly different colors -- than skin, hair, or
    most clothing, which makes this a useful *extra* candidate-generation
    signal alongside the brightness-based one below (neither alone is
    reliable on a busy real photo, which is exactly why we generate
    candidates from both and let the color-match check pick the winner)."""
    img = bgr_img.astype(np.float64)
    mean = cv2.boxFilter(img, ddepth=-1, ksize=(win, win))
    sq_mean = cv2.boxFilter(img * img, ddepth=-1, ksize=(win, win))
    var = np.clip(sq_mean - mean * mean, 0, None)
    return var.sum(axis=2)


def _saturated_hue_mask(bgr_img: np.ndarray, sat_min: int = 90, val_range=(40, 245), hue_tol: int = 14) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    base = (S > sat_min) & (V > val_range[0]) & (V < val_range[1])
    hit = np.zeros(base.shape, dtype=bool)
    for target in _TARGET_HUES:
        d = np.minimum(np.abs(H - target), 180 - np.abs(H - target))
        hit |= d < hue_tol
    return (base & hit).astype(np.uint8) * 255


def _small_squares_from_mask(mask: np.ndarray, img_area: int):
    """Individual small squarish blobs (single-patch scale) in a mask --
    deliberately NOT closed/merged into one big blob first, so a patch
    grid shows up as many separate small boxes rather than needing to
    already be one clean connected region."""
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.0008 or area > img_area * 0.04:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / float(bh)
        if 0.4 < ar < 2.5:
            boxes.append((x, y, bw, bh))
    return boxes


def _largest_grid_cluster(boxes, spacing_factor: float = 2.4, min_members: int = 6):
    """Group candidate squares by mutual proximity (union-find over a
    'closer than ~2.4 patch-widths' graph) and return the largest group.
    The card's 24 patches sit close together on a regular grid, so they
    form one dense cluster; a stray hit elsewhere (an eye, a ring, a
    fleck on clothing) sits far from the others and ends up alone or in
    a tiny cluster, which this discards."""
    if len(boxes) < min_members:
        return None
    centers = np.array([(x + bw / 2, y + bh / 2) for x, y, bw, bh in boxes])
    sizes = np.array([(bw + bh) / 2 for x, y, bw, bh in boxes])
    thresh = float(np.median(sizes)) * spacing_factor

    n = len(centers)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(centers[i] - centers[j]) < thresh:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    best_group = max(groups.values(), key=len)
    if len(best_group) < min_members:
        return None
    return [boxes[i] for i in best_group], float(np.median(sizes))


def _hue_grid_candidates(bgr_img: np.ndarray, pad_factors=(0.5, 0.8, 1.1, 1.4)):
    """Find the card by looking directly for its own known vivid colors
    (see _SATURATED_PATCH_NAMES) arranged in a tight, regular grid --
    this is the most specific signal available (skin/hair/most clothing
    simply don't have 6+ small patches of these exact hues sitting right
    next to each other on a grid) and is what lets automatic detection
    work even with dark hair or patterned clothing right next to the
    card. Runs on a downscaled copy for speed/parameter-stability, then
    scales the result back up to bgr_img's own resolution.
    """
    h, w = bgr_img.shape[:2]
    work_scale = 900 / max(h, w)
    small = cv2.resize(bgr_img, None, fx=work_scale, fy=work_scale) if work_scale < 1 else bgr_img
    sh, sw = small.shape[:2]
    img_area = sh * sw

    mask = _saturated_hue_mask(small)
    boxes = _small_squares_from_mask(mask, img_area)
    clustered = _largest_grid_cluster(boxes)
    if clustered is None:
        return []
    cluster_boxes, median_size = clustered

    pts = []
    for x, y, bw, bh in cluster_boxes:
        pts.append((x, y))
        pts.append((x + bw, y + bh))
    pts = np.array(pts, dtype=np.float32)

    quads = []
    for pad_factor in pad_factors:
        pad = median_size * pad_factor
        rect = cv2.minAreaRect(pts)
        (cx, cy), (rw, rh), ang = rect
        padded_rect = ((cx, cy), (rw + 2 * pad, rh + 2 * pad), ang)
        box = cv2.boxPoints(padded_rect)
        quad = _order_quad_points(box)
        if work_scale < 1:
            quad = quad / work_scale
        quads.append(quad.astype(np.float32))
    return quads


def _candidate_quads(bgr_img: np.ndarray):
    """Gather a pool of candidate card quads from several independent
    segmentation strategies. Intentionally over-generates (and tolerates
    duplicates/near-duplicates -- calibrate_from_image dedupes by trying
    each and skipping ones whose center is very close to an already-tried
    one) since the color-matching check downstream is what actually
    decides which candidate is the real card.
    """
    h, w = bgr_img.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    all_candidates = []

    # Strategy D (tried first -- it's the most specific signal): the
    # card's own known vivid colors, arranged in a tight regular grid.
    all_candidates.extend(_hue_grid_candidates(bgr_img))

    # Strategy A: Otsu global threshold (dark card vs bright background),
    # a couple of morphology strengths.
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    for ck, ok in ((9, 5), (5, 3), (13, 9)):
        m = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((ok, ok), np.uint8))
        all_candidates.extend(_quad_candidates_from_mask(m, img_area))

    # Strategy B: fixed brightness cutoffs (in case Otsu's split point is
    # skewed by a large dark background/hair).
    for cutoff in (40, 60, 80, 100):
        _, m = cv2.threshold(blur, cutoff, 255, cv2.THRESH_BINARY_INV)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        all_candidates.extend(_quad_candidates_from_mask(m, img_area))

    # Strategy C: local color-variance (catches the card even when it's
    # sitting right next to similarly-dark hair/background).
    var = _color_variance_map(bgr_img, win=15)
    denom = max(float(np.percentile(var, 99.5)), 1.0)
    var_norm = np.clip(var / denom * 255, 0, 255).astype(np.uint8)
    _, vm = cv2.threshold(var_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for ck, ok in ((9, 5), (13, 7)):
        m = cv2.morphologyEx(vm, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((ok, ok), np.uint8))
        all_candidates.extend(_quad_candidates_from_mask(m, img_area))

    return all_candidates


def find_card_quad(bgr_img: np.ndarray):
    """Backwards-compatible single-best-guess quad (kept for any external
    callers/tests); calibrate_from_image itself uses the full candidate
    pool + color-match validation instead of this."""
    candidates = _candidate_quads(bgr_img)
    return candidates[0] if candidates else None


def _warp_card_with_transform(bgr_img: np.ndarray, quad: np.ndarray):
    """Same as warp_card, but also returns the perspective matrix and
    canonical output size -- needed by _refine_quad_from_patches below to
    map a correction back the other way (canonical -> original image)."""
    side_w = max(
        np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])
    )
    side_h = max(
        np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])
    )
    landscape = side_w >= side_h
    out_w, out_h = (CANONICAL_W, CANONICAL_H) if landscape else (CANONICAL_H, CANONICAL_W)

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(bgr_img, M, (out_w, out_h))
    return warped, M, out_w, out_h


def warp_card(bgr_img: np.ndarray, quad: np.ndarray):
    warped, _M, _out_w, _out_h = _warp_card_with_transform(bgr_img, quad)
    return warped


def _refine_quad_from_patches(quad, boxes, M, out_w, out_h, orig_shape, border_units: float = -0.05):
    """The initial candidate quad (from _candidate_quads) is only a rough
    guess -- e.g. the hue-grid strategy pads a tight cluster of detected
    color patches outward by an assumed, fixed proportion to estimate
    where the card's actual edge is, which can overshoot (even past the
    photo's own edge on some tilts/angles) or undershoot depending on how
    the card sits in the frame, and can leave the card still sitting at a
    slight angle in the warped preview instead of flat.

    Picking any single corner (or 4 corners) of the grid to anchor a
    refit is fragile: the corner cells are exactly the ones most likely
    to be inferred/extrapolated rather than directly detected (dark
    patches like black_20 aren't picked up by the bright-patch contour
    step at all), so their individual position error is the highest of
    any cell in the grid -- basing the whole refit on just those 4 points
    means inheriting their worst-case error. Using all 24 detected patch
    centers is far more robust: we know their IDEAL positions exactly (a
    perfect 6x4 grid of unit cells, since that's the ColorChecker
    Classic's fixed physical layout) and can fit a single homography
    mapping that ideal grid to where the centers actually landed in this
    warp -- a 24-point least-squares fit averages out any one cell's
    noise instead of being dictated by it.

    `border_units` (in the same unit-cell scale) is deliberately small
    and slightly NEGATIVE by default: trying to extrapolate all the way
    out to the card's true physical edge (past the patches, out to its
    printed black border) turned out to be unreliable in practice -- any
    real camera has a little lens distortion a single homography can't
    model, and that error only grows when extrapolated beyond the fitted
    points, which is exactly what produced the persistent sliver of
    background this replaces. Landing slightly INSIDE the outermost
    patches instead is a much safer bet: it can never expose background,
    since it never leaves the region we actually, directly observed to
    be patch color, at the cost of not showing the card's own printed
    border in the preview -- a fine trade since only the 24 patch colors
    need to stay legible, not the card's physical frame. Mapping those 4
    points back through the INVERSE of this warp into the original
    image, then re-warping with them, "pulls"/stretches the patches to
    fill the whole canonical frame in one shot.
    """
    n = len(boxes)
    n_cols, n_rows = (6, 4) if out_w >= out_h else (4, 6)

    if n != n_cols * n_rows:
        # Not the expected clean grid -- fall back to a plain padded
        # bounding box (still an improvement over the original rough quad).
        xs0 = [x for (x, y, bw, bh) in boxes]
        ys0 = [y for (x, y, bw, bh) in boxes]
        xs1 = [x + bw for (x, y, bw, bh) in boxes]
        ys1 = [y + bh for (x, y, bw, bh) in boxes]
        median_w = float(np.median([bw for (_, _, bw, _bh) in boxes]))
        median_h = float(np.median([bh for (_, _, _bw, bh) in boxes]))
        margin_x, margin_y = median_w * 0.25, median_h * 0.25
        x0 = max(0.0, min(xs0) - margin_x)
        y0 = max(0.0, min(ys0) - margin_y)
        x1 = min(float(out_w), max(xs1) + margin_x)
        y1 = min(float(out_h), max(ys1) + margin_y)
        canonical_corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    else:
        # boxes is in deterministic row-major grid order (see
        # detect_patches: `for cy in row_ys: for cx in col_xs: ...`).
        centers = np.array(
            [[x + bw / 2.0, y + bh / 2.0] for (x, y, bw, bh) in boxes], dtype=np.float64
        )
        ideal = np.array(
            [[c + 0.5, r + 0.5] for r in range(n_rows) for c in range(n_cols)],
            dtype=np.float64,
        )

        H, _mask = cv2.findHomography(ideal, centers, method=0)

        ideal_card_corners = np.array(
            [
                [-border_units, -border_units],
                [n_cols + border_units, -border_units],
                [n_cols + border_units, n_rows + border_units],
                [-border_units, n_rows + border_units],
            ],
            dtype=np.float64,
        )
        ones = np.ones((4, 1), dtype=np.float64)
        homog = np.hstack([ideal_card_corners, ones])
        canonical_mapped = (H @ homog.T).T
        canonical_corners = canonical_mapped[:, :2] / canonical_mapped[:, 2:3]

    canonical_corners[:, 0] = np.clip(canonical_corners[:, 0], 0, out_w)
    canonical_corners[:, 1] = np.clip(canonical_corners[:, 1], 0, out_h)

    M_inv = np.linalg.inv(M)
    ones = np.ones((4, 1), dtype=np.float64)
    homog = np.hstack([canonical_corners, ones])
    mapped = (M_inv @ homog.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]

    h_img, w_img = orig_shape[:2]
    mapped[:, 0] = np.clip(mapped[:, 0], 0, w_img - 1)
    mapped[:, 1] = np.clip(mapped[:, 1], 0, h_img - 1)

    return mapped.astype(np.float32)


def _warp_and_detect_refined(bgr_img: np.ndarray, quad: np.ndarray, n_refine_passes: int = 2):
    """Rough warp + patch-detect pass, then up to `n_refine_passes` rounds
    of _refine_quad_from_patches (see there for why one pass is needed at
    all). A tilted card can need more than one round to fully converge --
    each pass re-derives the quad from where the patches actually landed
    in the PREVIOUS pass's warp, so residual skew shrinks with each round
    instead of needing to be fully corrected in one shot. Stops early once
    a pass stops finding more patches (no more room to improve) or the
    quad barely moves. Always keeps the best (most-patches-found) result
    seen, so a bad later pass can never make things worse than an earlier
    good one. Returns (quad_used, warped_used, boxes_used)."""
    warped, M, out_w, out_h = _warp_card_with_transform(bgr_img, quad)
    boxes = detect_patches(warped)
    if len(boxes) < 20:
        return quad, warped, boxes

    best_quad, best_warped, best_boxes = quad, warped, boxes

    for _ in range(n_refine_passes):
        refined_quad = _refine_quad_from_patches(
            best_quad, best_boxes, M, out_w, out_h, bgr_img.shape
        )
        if np.allclose(refined_quad, best_quad, atol=1.0):
            break  # converged -- another pass wouldn't change anything
        refined_warped, M, out_w, out_h = _warp_card_with_transform(bgr_img, refined_quad)
        refined_boxes = detect_patches(refined_warped)
        if len(refined_boxes) < 20:
            break
        if len(refined_boxes) >= len(best_boxes):
            best_quad, best_warped, best_boxes = refined_quad, refined_warped, refined_boxes
        else:
            break

    return best_quad, best_warped, best_boxes


def _find_bright_patch_boxes(warped_bgr: np.ndarray):
    """Contour-detect the lighter/more saturated patches (dark ones such as
    black, dark gray, dark purple blend into the card's black grid and are
    reliably missed here -- that's fixed by _complete_grid below)."""
    h, w = warped_bgr.shape[:2]
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = h * w
    expected_patch_area = total_area / 24
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < expected_patch_area * 0.25 or area > expected_patch_area * 2.5:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / float(bh)
        if 0.5 < ar < 2.0:
            boxes.append((x, y, bw, bh))
    return boxes


def _kmeans_1d(values, k, n_iter=50):
    """Small fixed-k 1D k-means (grid column/row positions are well
    separated, so this converges trivially -- avoids depending on an
    extra clustering library)."""
    values = np.array(sorted(values), dtype=np.float64)
    # even-quantile init keeps it stable regardless of uneven point counts per cluster
    centers = np.quantile(values, np.linspace(0.05, 0.95, k))
    for _ in range(n_iter):
        dists = np.abs(values[:, None] - centers[None, :])
        assign = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            pts = values[assign == i]
            if len(pts) > 0:
                new_centers[i] = pts.mean()
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return sorted(centers.tolist())


def detect_patches(warped_bgr: np.ndarray):
    """Find all 24 color-patch cells inside a warped card image.

    Bright/saturated patches are found directly by contour. Their centers
    are then clustered into grid rows and columns (the card's patches are
    laid out on a perfectly regular grid), which reveals the row/column
    positions even for dark patches no contour was found for -- those
    cells are filled in at the expected grid intersection.
    """
    boxes = _find_bright_patch_boxes(warped_bgr)
    if len(boxes) < 6:
        return boxes  # too little signal to infer a grid

    h, w = warped_bgr.shape[:2]
    # The ColorChecker Classic is always 6x4 patches; we warp to a canonical
    # rectangle above so the orientation (landscape vs portrait) tells us
    # which axis has 6 and which has 4.
    n_cols, n_rows = (6, 4) if w >= h else (4, 6)

    centers = [(x + bw / 2, y + bh / 2) for (x, y, bw, bh) in boxes]
    median_w = float(np.median([b[2] for b in boxes]))
    median_h = float(np.median([b[3] for b in boxes]))

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    col_xs = _kmeans_1d(xs, n_cols)
    row_ys = _kmeans_1d(ys, n_rows)

    full_boxes = []
    for cy in row_ys:
        for cx in col_xs:
            full_boxes.append(
                (int(cx - median_w / 2), int(cy - median_h / 2), int(median_w), int(median_h))
            )
    return full_boxes


def _sample_patch_color(warped_bgr: np.ndarray, box):
    x, y, bw, bh = box
    # Sample the central 50% of the patch to avoid grid-line/edge bleed.
    cx0 = x + int(bw * 0.25)
    cx1 = x + int(bw * 0.75)
    cy0 = y + int(bh * 0.25)
    cy1 = y + int(bh * 0.75)
    crop = warped_bgr[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        crop = warped_bgr[y : y + bh, x : x + bw]
    median_bgr = np.median(crop.reshape(-1, 3), axis=0)
    return median_bgr[::-1]  # -> RGB


def match_patches_to_reference(observed_rgb: np.ndarray, reference_rgb_list=None):
    """Hungarian-match observed patch colors to the 24 known reference
    colors. reference_rgb_list defaults to the ACTIVE reference (a custom
    one captured from the admin page's "clean card photo" flow if one has
    been saved, else the standard official ColorChecker values) -- see
    reference_colors.get_active_reference_rgb_list(). Passing the default
    REFERENCE_RGB_LIST explicitly is only for the one place that must
    always use the official values regardless of any custom reference:
    capture_card_reference() below, when establishing a NEW custom
    reference in the first place."""
    if reference_rgb_list is None:
        reference_rgb_list = get_active_reference_rgb_list()
    ref = np.array(reference_rgb_list, dtype=np.float64)
    obs = np.array(observed_rgb, dtype=np.float64)
    n = min(len(ref), len(obs))
    cost = np.zeros((len(obs), len(ref)))
    for i, o in enumerate(obs):
        cost[i] = np.linalg.norm(ref - o, axis=1)
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind, cost


def solve_correction_matrix(observed_rgb: np.ndarray, reference_rgb: np.ndarray):
    """Least-squares affine map: reference ≈ M @ [observed; 1]."""
    obs = np.array(observed_rgb, dtype=np.float64)
    ref = np.array(reference_rgb, dtype=np.float64)
    ones = np.ones((obs.shape[0], 1))
    A = np.hstack([obs, ones])  # N x 4
    M, *_ = np.linalg.lstsq(A, ref, rcond=None)  # 4 x 3
    return M.T  # 3 x 4


def apply_correction(rgb: np.ndarray, M: np.ndarray) -> np.ndarray:
    rgb = np.array(rgb, dtype=np.float64)
    vec = np.append(rgb, 1.0)
    corrected = M @ vec
    return np.clip(corrected, 0, 255)


def apply_correction_image(rgb_img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Same affine correction as apply_correction, but vectorized over a
    whole HxWx3 RGB image at once (for a debug before/after preview) --
    equivalent to calling apply_correction on every pixel, just fast."""
    M = np.array(M, dtype=np.float64)
    img = np.array(rgb_img, dtype=np.float64)
    # corrected = M[:, :3] @ pixel + M[:, 3], applied to every pixel
    corrected = img @ M[:, :3].T + M[:, 3]
    return np.clip(corrected, 0, 255).astype(np.uint8)


# How much of the full card-based correction to actually apply, from 0.0
# (no correction at all) to 1.0 (the full, mathematically "correct"
# camera/lighting correction). The reference app ("제루미") the product
# owner is matching against visibly applies a much milder correction than
# the full physically-derived one -- e.g. for one test photo, the full
# correction moves the skin color by ΔE00≈10.3, while the reference app's
# own before/after only moves it by ΔE00≈2.5. Blending the fitted matrix
# toward the identity transform at strength≈0.3 reproduces that same,
# gentler magnitude (~ΔE00 3.0 on the same test case) while still pointing
# in the same (camera-bias-correcting) direction. This is a deliberate,
# tunable product choice, not a bug fix -- turning it up trades a more
# "technically correct" (but visually stronger) correction for matching
# the reference app's subtler look; turning it down or up is just editing
# this one constant.
# [2026-10 변경] 보정을 "색깔"과 "밝기" 두 가지로 나눠서 따로 강도를 정해요.
#  - 색깔(CORRECTION_STRENGTH): 사진에 낀 푸른 기·붉은 기 같은 색 치우침을 얼마나 바로잡을지.
#    약하게 하면(예: 0.12) 푸른 기가 남아서 피부가 회분홍색으로 탁해져요.
#    카드의 피부색 계열 칸으로 검증해 보니 0.7~1.0 구간이 가장 정확했고, 그중 안전하게 0.7로 정했어요. (0.0 = 안 함, 1.0 = 카드 기준 그대로)
#  - 밝기(BRIGHTNESS_CORRECTION_STRENGTH): 카드 기준으로 얼마나 밝게/어둡게 바꿀지.
#    카드가 얼굴과 다른 빛을 받는 경우가 많아서, 크게 바꾸면 피부가 어둡고 칙칙해져요 -> 약하게(12%).
CORRECTION_STRENGTH = 0.7
BRIGHTNESS_CORRECTION_STRENGTH = 0.12


def _lab_to_rgb255(lab) -> np.ndarray:
    """CIELAB(D65) -> sRGB 0~255 (color_match.rgb2lab_01의 반대 방향)."""
    L, a, b = [float(v) for v in lab]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    delta = 6.0 / 29.0

    def finv(t):
        return t ** 3 if t > delta else 3 * delta * delta * (t - 4.0 / 29.0)

    xyz = np.array([0.95047 * finv(fx), 1.0 * finv(fy), 1.08883 * finv(fz)])
    m_inv = np.linalg.inv(
        np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )
    )
    lin = np.clip(m_inv @ xyz, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(srgb * 255.0, 0, 255)


def correct_rgb(raw_rgb, M: np.ndarray, brightness_strength: float = None) -> np.ndarray:
    """카드 보정을 적용한 피부색(sRGB 0~255)을 돌려줘요.

    - 색깔(a*, b*): 보정 행렬을 그대로 적용한 값을 써요 (푸른 기·붉은 기를 바로잡아요).
    - 밝기(L*): 원래 사진 밝기에서 보정이 바꾸려는 만큼의 BRIGHTNESS_CORRECTION_STRENGTH 비율만 반영해요.
      (카드가 얼굴과 다른 빛을 받으면 밝기 보정이 피부를 어둡고 탁하게 만들기 때문이에요.)
    """
    from .color_match import rgb_to_lab

    k = BRIGHTNESS_CORRECTION_STRENGTH if brightness_strength is None else brightness_strength
    raw = np.array(raw_rgb, dtype=np.float64)
    corrected = apply_correction(raw, M)
    lab_raw = rgb_to_lab(raw)
    lab_cor = rgb_to_lab(corrected)
    L = float(lab_raw[0] + k * (lab_cor[0] - lab_raw[0]))
    return _lab_to_rgb255([L, lab_cor[1], lab_cor[2]])


def _try_calibrate_candidate(bgr_img: np.ndarray, quad: np.ndarray):
    """Run the warp -> patch-detect -> color-match -> solve pipeline for
    ONE candidate quad. Returns a CalibrationResult (success may be False
    if this particular candidate didn't pan out -- that's expected for
    most candidates; calibrate_from_image tries several and keeps the
    best)."""
    quad, warped, boxes = _warp_and_detect_refined(bgr_img, quad)
    if len(boxes) < 20:
        return CalibrationResult(False, f"카드 안 색상 패치를 충분히 찾지 못했어요 ({len(boxes)}/24개 인식)")

    observed_rgb = np.array([_sample_patch_color(warped, b) for b in boxes])
    if not np.all(np.isfinite(observed_rgb)):
        # 카드 가장자리 때문에 비어 있는 칸이 생기면 NaN이 나와서 계산이 터졌어요 -> 이 후보는 건너뛰어요.
        return CalibrationResult(False, "카드 칸 색을 읽지 못한 후보예요")
    active_reference = get_active_reference_rgb_list()
    row_ind, col_ind, _cost = match_patches_to_reference(observed_rgb, active_reference)

    matched_observed = observed_rgb[row_ind]
    matched_reference = np.array(active_reference)[col_ind]

    # A patch with a channel pinned at (near) 0 or 255 is clipped/blown out
    # (usually the white patch catching a specular highlight) -- its
    # "observed" color isn't real, so fitting the correction matrix to it
    # distorts the whole transform. Fit on the non-clipped patches only,
    # but fall back to using everything if too many are clipped (e.g. a
    # very harshly lit photo) so we don't end up with too few equations.
    clip_lo, clip_hi = 3, 252
    not_clipped = ~np.any((matched_observed <= clip_lo) | (matched_observed >= clip_hi), axis=1)
    fit_observed = matched_observed[not_clipped] if not_clipped.sum() >= 12 else matched_observed
    fit_reference = matched_reference[not_clipped] if not_clipped.sum() >= 12 else matched_reference

    M = solve_correction_matrix(fit_observed, fit_reference)

    # mean_err (and thus which candidate quad "wins" in calibrate_from_image)
    # is deliberately measured against the FULL-strength matrix, not the
    # blended one below -- it's a measure of how well the card itself was
    # detected/read, which shouldn't change just because we've since decided
    # to apply a gentler correction to the final photo.
    corrected = np.array([apply_correction(o, M) for o in matched_observed])
    mean_err = float(np.mean(np.linalg.norm(corrected - matched_reference, axis=1)))

    # Blend the fitted correction toward "do nothing" (identity) so the
    # correction actually applied to the photo is milder than the full,
    # physically-derived one -- see CORRECTION_STRENGTH above.
    M_identity = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    M_applied = CORRECTION_STRENGTH * M + (1 - CORRECTION_STRENGTH) * M_identity

    centers = np.array([[x + bw / 2, y + bh / 2] for (x, y, bw, bh) in boxes])

    return CalibrationResult(
        True,
        f"카드 {len(boxes)}개 패치 인식 완료, 평균 보정 오차 {mean_err:.1f}",
        correction_matrix=M_applied,
        mean_delta_e=mean_err,
        card_corners=quad,
        patch_centers_canonical=centers[row_ind],
        observed_patch_rgb=matched_observed,
        matched_reference_rgb=matched_reference,
    )


def capture_card_reference(bgr_img: np.ndarray):
    """For the admin "카드 기준값 재설정" flow: given ONE clean, well-lit,
    straight-on photo of just the physical card (no face, no glare), find
    it and return its 24 patch colors AS OBSERVED -- no correction
    applied. These raw observed values are meant to be saved as the new
    correction TARGET (see reference_colors.get_active_reference_patches /
    github_storage.write_card_reference), replacing the standard official
    Calibrite/X-Rite numbers with what this specific physical card and
    camera combination actually reads under good light. A printed card
    can drift a bit from the published spec (printing batch, aging), so
    this can be more accurate for THIS card than the official table.

    Patches are identified (which detected square is "orange_yellow" vs
    "light_skin" etc.) by nearest-match to the STANDARD official
    reference colors, regardless of any custom reference already active
    -- that's only for figuring out which patch is which, not for the
    value that gets stored.

    Requires all 24 patches to be found (stricter than the normal ~20/24
    tolerance used on real face photos) since the whole point is a clean,
    reliable baseline. Returns (success, message, patches_dict_or_None)
    where patches_dict is {name: [R, G, B]}.
    """
    candidates = _candidate_quads(bgr_img)
    if not candidates:
        return False, "카드를 사진에서 찾지 못했어요 — 카드만 크고 선명하게 나오도록 다시 찍어주세요", None

    best = None
    for quad in candidates:
        quad, warped, boxes = _warp_and_detect_refined(bgr_img, quad)
        if len(boxes) < 20:
            continue
        observed_rgb = np.array([_sample_patch_color(warped, b) for b in boxes])
        row_ind, col_ind, cost = match_patches_to_reference(observed_rgb, REFERENCE_RGB_LIST)
        total_cost = float(cost[row_ind, col_ind].sum())
        if best is None or total_cost < best[0]:
            best = (total_cost, row_ind, col_ind, observed_rgb)

    if best is None:
        return False, "카드 패치를 충분히 찾지 못했어요 — 반사·그림자 없이 카드가 꽉 차게 다시 찍어주세요", None

    _cost, row_ind, col_ind, observed_rgb = best
    if len(row_ind) < 24:
        return False, f"24개 중 {len(row_ind)}개 패치만 인식됐어요 — 카드 전체가 잘리지 않게 다시 찍어주세요", None

    patches = {
        REFERENCE_NAMES[c]: [round(float(x), 1) for x in observed_rgb[r]] for r, c in zip(row_ind, col_ind)
    }
    return True, "카드 24개 패치 전부 인식 완료", patches


# A candidate whose best achievable color-matching error is above this is
# treated as "not actually the card" (e.g. a patch of striped shirt or
# jewelry that happened to look card-shaped), even if it was the geometric
# front-runner. The real card, even under fairly rough lighting, comes in
# well under this.
MAX_ACCEPTABLE_MEAN_ERROR = 55.0


def _best_over_candidates(bgr_img: np.ndarray, finder_img: np.ndarray):
    """finder_img에서 카드 후보 위치를 찾고, 각 후보를 bgr_img(원본)로 보정해서 가장 잘 맞는 결과를 돌려줘요.
    (finder_img는 원본과 크기가 같고, 밝기만 다를 수 있어요 -- 위치는 그대로 쓸 수 있어요.)
    후보 하나가 계산 중 에러를 내도 다음 후보로 넘어가요."""
    candidates = _candidate_quads(finder_img)
    if not candidates:
        return None, 0

    h, w = bgr_img.shape[:2]
    dedupe_radius = 0.03 * max(h, w)
    # Dedupe on (center, size) together -- two quads can share a center but
    # be genuinely different candidates (e.g. the same hue-grid cluster
    # padded by different amounts), so center proximity alone must not
    # skip one of them.
    tried = []  # list of (center, diag)
    best = None

    for quad in candidates:
        center = quad.mean(axis=0)
        diag = float(np.linalg.norm(quad[2] - quad[0]))  # corner-to-corner size
        is_dup = any(
            np.linalg.norm(center - c) < dedupe_radius and abs(diag - d) < 0.15 * max(diag, d)
            for c, d in tried
        )
        if is_dup:
            continue
        tried.append((center, diag))

        try:
            result = _try_calibrate_candidate(bgr_img, quad)
        except Exception:
            continue
        if not result.success:
            continue
        if best is None or result.mean_delta_e < best.mean_delta_e:
            best = result
    return best, len(tried)


# 카드 위치를 못 찾았을 때, 사진을 이 정도로 어둡게 만든 복사본으로 위치만 다시 찾아봐요.
# (너무 밝게 찍혀서 카드의 밝은 색 칸이 하얗게 날아간 사진도 어둡게 하면 색 칸이 살아나요.)
# 색 보정 계산 자체는 항상 원본 사진으로 해요.
_DARKEN_FALLBACK_GAMMAS = (1.3, 1.6, 2.0)


def _darken(bgr_img: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.array([255.0 * ((i / 255.0) ** gamma) for i in range(256)]).clip(0, 255).astype(np.uint8)
    return cv2.LUT(bgr_img, lut)


def calibrate_from_image(bgr_img: np.ndarray) -> CalibrationResult:
    """Automatically find and calibrate against the ColorChecker card.

    Rather than trusting a single "most likely" card-shaped region, this
    tries every plausible candidate region from `_candidate_quads` and
    keeps whichever one actually matches the card's known reference
    colors best -- since the 24 reference colors are fixed and known in
    advance, "how well do this region's colors match them" is a much
    stronger and more specific test than any purely geometric guess, and
    is what lets this stay fully automatic even when hair, clothing
    patterns, or jewelry are right next to the card in the photo.

    카드를 못 찾으면(주로 사진이 너무 밝아 색 칸이 날아간 경우) 어둡게 만든 복사본으로
    위치만 다시 찾아본 뒤, 보정은 원본으로 해요. 먼저 성공한 사진의 결과는 예전과 같아요.
    """
    best, n_tried = _best_over_candidates(bgr_img, bgr_img)

    if best is None or best.mean_delta_e > MAX_ACCEPTABLE_MEAN_ERROR:
        for g in _DARKEN_FALLBACK_GAMMAS:
            alt, n_alt = _best_over_candidates(bgr_img, _darken(bgr_img, g))
            n_tried += n_alt
            if alt is not None and (best is None or alt.mean_delta_e < best.mean_delta_e):
                best = alt
            if best is not None and best.mean_delta_e <= MAX_ACCEPTABLE_MEAN_ERROR:
                break

    if best is None or best.mean_delta_e > MAX_ACCEPTABLE_MEAN_ERROR:
        return CalibrationResult(
            False,
            "카드를 사진에서 찾지 못했어요 (색상카드가 잘 보이게, 너무 밝거나 어둡지 않은 곳에서 다시 촬영해주세요)",
        )

    best.n_candidates_tried = n_tried
    return best
