"""
End-to-end pipeline: a single photo containing BOTH the ColorChecker card
and the subject's face -> camera/lighting-corrected skin Lab value ->
best-matching foundation shade(s) from the DB.

Skin color step: sample both lower cheeks rather than a single point,
correct each with the card-based calibration, convert to Lab, then take
whichever reads lightest (highest L) as the final skin tone -- shadow
only ever darkens a reading relative to its true color, so the lightest
of the sampled regions is, by construction, the least shadow-affected.
"""
from dataclasses import dataclass, field
import numpy as np

from .calibration import calibrate_from_image, correct_rgb
from .skin_extraction import extract_skin_regions
from .color_match import rgb_to_lab, find_best_matches_meta, delta_e_ciede2000


@dataclass
class RecommendationResult:
    success: bool
    message: str = ""
    corrected_lab: np.ndarray = None
    matches: list = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def _pairwise_delta_e(lab_list) -> np.ndarray:
    n = len(lab_list)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            lab_i = np.array(lab_list[i], dtype=np.float64).reshape(1, 1, 3)
            lab_j = np.array(lab_list[j], dtype=np.float64).reshape(1, 1, 3)
            de = float(delta_e_ciede2000(lab_i, lab_j)[0, 0])
            d[i, j] = d[j, i] = de
    return d


def _agreement_clusters(dist: np.ndarray, threshold: float):
    """Union-find over region indices: connect i,j whenever their
    perceptual distance is within `threshold` (i.e. they plausibly read
    the same real skin tone). Returns a list of clusters (each a list of
    indices)."""
    n = dist.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _select_medoid(region_labs, agreement_threshold: float = 8.0, region_names=None):
    """Select the ROI with the highest L (lightness) value -- i.e. the
    brightest-reading region -- and use it as-is.

    Earlier versions of this tried to statistically detect and drop
    shadow-contaminated regions (union-find clustering on mutual
    agreement, then an anatomical cheek-priority rule). Both still
    occasionally let a shadowed region (chin/under_mouth, which sit next
    to the same shadow-casting features and can look artificially
    "consistent" with each other) win over the true, better-lit skin
    tone. Per explicit product decision, this is replaced with a much
    simpler and more robust rule: shadow only ever DARKENS a region
    relative to its true color, never lightens it -- so whichever ROI
    comes out lightest is, by construction, the least shadow-affected
    reading, without needing any clustering/tiebreak logic at all.

    Returns (final_lab, medoid_index, kept_indices, dist_matrix)."""
    n = len(region_labs)
    dist = _pairwise_delta_e(region_labs) if n > 1 else np.zeros((1, 1))

    l_values = [lab[0] for lab in region_labs]
    medoid_idx = int(np.argmax(l_values))
    kept = [medoid_idx]

    return region_labs[medoid_idx], medoid_idx, kept, dist


def recommend_foundation(
    bgr_img: np.ndarray, shades: list, top_n: int = 3, card_bgr: np.ndarray = None,
    has_card: bool = True,
) -> RecommendationResult:
    """bgr_img: the full photo (card + face), used for face/skin detection.
    card_bgr: optional -- a crop containing ONLY the color card, used for
    calibration instead of bgr_img. Real photos with hair/clothing/jewelry
    next to the card can fool automatic card-quad detection, so the card
    region is user-cropped in the UI; pass that crop here. Falls back to
    bgr_img itself (old automatic-detection-on-the-whole-photo behavior)
    when not given.

    has_card: False is the path for someone who doesn't own a physical
    ColorChecker card. There's no known-color reference in the photo to
    measure the camera/lighting's color bias against, so card detection
    and correction are skipped entirely -- the raw sampled cheek color is
    used as the "corrected" color as-is. This is deliberately less
    accurate (no camera/lighting bias removal at all), but lets someone
    without the card still get a recommendation rather than being turned
    away.
    """
    calib = None
    if has_card:
        calib = calibrate_from_image(card_bgr if card_bgr is not None else bgr_img)
        if not calib.success:
            return RecommendationResult(False, f"색상카드 인식 실패: {calib.message}")

    skin = extract_skin_regions(bgr_img)
    if not skin.success:
        return RecommendationResult(False, f"얼굴 인식 실패: {skin.message}")

    region_labs = []
    region_debug = []
    for region in skin.regions:
        if has_card:
            corrected_rgb = correct_rgb(region.raw_rgb, calib.correction_matrix)
        else:
            corrected_rgb = np.array(region.raw_rgb, dtype=np.float64)
        lab = rgb_to_lab(corrected_rgb)
        region_labs.append(lab)
        region_debug.append(
            {
                "name": region.name,
                "raw_rgb": [round(float(x), 1) for x in region.raw_rgb],
                "corrected_rgb": [round(float(x), 1) for x in corrected_rgb],
                "lab": [round(float(x), 2) for x in lab],
                "center": region.center,
                "radius": region.radius,
                "n_pixels_total": region.n_pixels_total,
                "n_pixels_used": region.n_pixels_used,
                "clip_pct": region.clip_pct,
                "lum_low": region.lum_low,
                "lum_high": region.lum_high,
            }
        )

    final_lab, medoid_idx, kept_idx, _dist = _select_medoid(
        region_labs, region_names=[r.name for r in skin.regions]
    )
    for i, rd in enumerate(region_debug):
        rd["used_as_final"] = i == medoid_idx
        rd["excluded_as_outlier"] = i not in kept_idx

    matches, match_meta = find_best_matches_meta(final_lab, shades, top_n=top_n, require_brighter=True)

    return RecommendationResult(
        True,
        "추천 완료",
        corrected_lab=final_lab,
        matches=matches,
        debug={
            "regions": region_debug,
            "final_region": skin.regions[medoid_idx].name,
            "image_size": list(skin.image_size) if skin.image_size else None,
            "has_card": has_card,
            "calibration_mean_error": calib.mean_delta_e if calib is not None else None,
            "correction_matrix": calib.correction_matrix.tolist() if calib is not None else None,
            "brighter_pool_used": match_meta["brighter_pool_used"],
            "n_candidates_considered": match_meta["n_candidates"],
        },
    )


def build_shade_from_photos(bgr_imgs: list, swatch_box_picker=None) -> dict:
    """For the admin DB-builder: given up to 5 photos of (card + foundation
    swatch), calibrate each and sample the swatch region, then average.

    swatch_box_picker(calib, bgr_img) -> (x, y, w, h) in original image
    coordinates for where the swatch is; if not provided, the caller is
    expected to have already cropped each image to just the swatch area
    next to the card (simplest for the admin UI: user marks/crops it).
    Returns {"L":.., "a":.., "b":.., "n_photos_used":..}
    """
    labs = []
    for img in bgr_imgs:
        calib = calibrate_from_image(img)
        if not calib.success:
            continue
        if swatch_box_picker is not None:
            box = swatch_box_picker(calib, img)
            x, y, w, h = box
            crop = img[y : y + h, x : x + w]
        else:
            crop = img
        median_bgr = np.median(crop.reshape(-1, 3), axis=0)
        raw_rgb = median_bgr[::-1]
        corrected_rgb = correct_rgb(raw_rgb, calib.correction_matrix)
        labs.append(rgb_to_lab(corrected_rgb))

    if not labs:
        return None
    mean_lab = np.mean(labs, axis=0)
    return {"L": round(float(mean_lab[0]), 3), "a": round(float(mean_lab[1]), 3), "b": round(float(mean_lab[2]), 3), "n_photos_used": len(labs)}
