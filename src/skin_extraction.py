"""
Face landmark based skin-tone sampling.

Multi-region ROI method: samples both lower cheeks, which tend to avoid
eyebrows/eyelashes/hair and the shadow that often falls under the eyes,
nose, and chin. Within each region, the brightest and darkest pixels
(specular highlight / shadow) are dropped before taking the median
color, matching the report's "클리핑·그림자 픽셀 제거" step.

(Earlier drafts also sampled under_mouth and chin, matching the original
team's 하부 볼 / 입 아래 / 턱 documented pipeline. Those two were removed
per product decision: they sit low on the face, right next to the same
shadow-casting features -- the lower lip, the jawline -- so they tend to
read darker than the cheeks and don't add anything the cheeks don't
already cover better.)

This module only returns the raw (uncorrected) color per region. The
camera-color correction (from calibration.py) and the final
brightest-region selection across regions happen in recommend.py, since
that is where the corrected Lab values are available.

Landmark indices (mediapipe FaceLandmarker's 478-point face mesh -- same
topology/index numbering as the older FaceMesh(refine_landmarks=True)):
  50, 280   - a point on each cheek, below the eye and above the mouth
              corner (commonly used "cheek" landmarks in AR/makeup apps)
  468, 473  - left/right iris centers -- used only to scale the ROI
              radius to the person's actual face size via interpupillary
              distance.

NOTE on the mediapipe API: as of mediapipe 0.10.2x+, the old
`mp.solutions.face_mesh` API used in earlier drafts of this file no
longer exists in the pip package at all (confirmed against the installed
0.10.32 -- `mediapipe.solutions` is gone, not just deprecated). The
current, supported way to get face landmarks is the newer "Tasks" API
(`mediapipe.tasks.python.vision.FaceLandmarker`), which needs a small
model file (`face_landmarker.task`, a few MB) bundled alongside the code
instead of a model baked into the pip package. See MODEL_PATH below --
this file must exist in the deployed repo at `models/face_landmarker.task`
(download link + instructions given separately) or face detection will
fail with a clear "모델 파일을 찾을 수 없어요" message rather than a crash.
"""

import os
from dataclasses import dataclass, field
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import core as mp_core

LEFT_IRIS_CENTER_IDX = 468
RIGHT_IRIS_CENTER_IDX = 473

# name -> landmark index
REGION_LANDMARKS = {
    "cheek_a": 50,   # cheek, camera-frame side A
    "cheek_b": 280,  # cheek, camera-frame side B
}

# repo_root/models/face_landmarker.task (this file lives in repo_root/src/)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_landmarker.task")

_landmarker = None
_landmarker_init_error = None


def _get_landmarker():
    global _landmarker, _landmarker_init_error
    if _landmarker is not None or _landmarker_init_error is not None:
        return _landmarker
    if not os.path.exists(MODEL_PATH):
        _landmarker_init_error = f"얼굴 인식 모델 파일이 없어요 ({MODEL_PATH})"
        return None
    try:
        # Force CPU delegate: Streamlit Cloud's servers have no GPU, and
        # mediapipe's GPU delegate pulls in EGL/OpenGL shared libraries
        # (libEGL.so.1) that aren't installed there by default -- explicitly
        # requesting CPU avoids needing those libraries at all.
        base_options = mp_core.base_options.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=mp_core.base_options.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    except Exception as e:  # pragma: no cover - defensive, surfaced to the UI
        _landmarker_init_error = f"얼굴 인식 모델을 불러오지 못했어요 ({e})"
    return _landmarker


@dataclass
class SkinRegionSample:
    name: str
    raw_rgb: np.ndarray
    center: tuple
    radius: int
    n_pixels_used: int
    n_pixels_total: int = 0          # 원 안에 들어온 전체 픽셀 수
    clip_pct: float = 15.0           # 밝기 상·하위 몇 %를 제외했는지
    lum_low: float = 0.0             # 이 밝기보다 어두우면 제외(그림자 쪽)
    lum_high: float = 0.0            # 이 밝기보다 밝으면 제외(빛 반사 쪽)


@dataclass
class SkinSampleResult:
    success: bool
    message: str = ""
    regions: list = field(default_factory=list)  # list[SkinRegionSample]
    interpupillary_dist: float = None
    image_size: tuple = None  # (width, height) -- 화면에 영역을 그릴 때 비율 계산용


def _landmark_px(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def _sample_region(rgb_img: np.ndarray, center, radius: int, clip_pct: float = 15.0):
    """Median color within a circular ROI, after dropping the brightest
    and darkest `clip_pct` percent of pixels by luminance (removes
    specular-highlight and shadow pixels rather than blending them into
    the average). Returns (median_rgb, n_pixels_kept) or None if the ROI
    falls outside the image.

    NOTE: 결과 화면에서 "어떻게 걸러냈는지"를 보여주기 위해 전체 픽셀 수와
    제외 기준 밝기값(lo/hi)도 함께 돌려줘요."""
    h, w = rgb_img.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    if not (0 <= cx < w and 0 <= cy < h):
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    pixels = rgb_img[mask == 255].reshape(-1, 3).astype(np.float64)
    if pixels.shape[0] == 0:
        return None

    luminance = pixels @ np.array([0.299, 0.587, 0.114])
    lo, hi = np.percentile(luminance, [clip_pct, 100 - clip_pct])
    keep = (luminance >= lo) & (luminance <= hi)
    kept = pixels[keep] if np.any(keep) else pixels

    median_rgb = np.median(kept, axis=0)
    return median_rgb, int(kept.shape[0]), int(pixels.shape[0]), float(lo), float(hi), float(clip_pct)


def extract_skin_regions(bgr_img: np.ndarray, radius_factor: float = 0.10) -> SkinSampleResult:
    """Detect the face and sample raw color at each of the 4 ROIs above.
    radius_factor scales with interpupillary distance so the ROI size
    adapts to how close/far the face is in the photo."""
    landmarker = _get_landmarker()
    if landmarker is None:
        return SkinSampleResult(False, _landmarker_init_error or "얼굴 인식 모델을 불러오지 못했어요")

    h, w = bgr_img.shape[:2]
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return SkinSampleResult(False, "사진에서 얼굴을 찾지 못했어요")

    landmarks = result.face_landmarks[0]

    iris_a = _landmark_px(landmarks, LEFT_IRIS_CENTER_IDX, w, h)
    iris_b = _landmark_px(landmarks, RIGHT_IRIS_CENTER_IDX, w, h)
    interpupillary_dist = float(np.linalg.norm(iris_a - iris_b))
    radius = max(5, int(interpupillary_dist * radius_factor))

    regions = []
    for name, idx in REGION_LANDMARKS.items():
        center = _landmark_px(landmarks, idx, w, h)
        sampled = _sample_region(rgb, center, radius)
        if sampled is None:
            continue
        median_rgb, n_used, n_total, lum_lo, lum_hi, clip_pct = sampled
        regions.append(
            SkinRegionSample(
                name=name,
                raw_rgb=median_rgb,
                center=(int(center[0]), int(center[1])),
                radius=radius,
                n_pixels_used=n_used,
                n_pixels_total=n_total,
                clip_pct=clip_pct,
                lum_low=round(lum_lo, 1),
                lum_high=round(lum_hi, 1),
            )
        )

    if not regions:
        return SkinSampleResult(
            False,
            "피부색 샘플링 영역이 사진 범위를 벗어났어요 (얼굴이 더 잘 보이게 다시 촬영해주세요)",
        )

    return SkinSampleResult(
        True, "피부색 샘플 추출 완료", regions=regions, interpupillary_dist=interpupillary_dist,
        image_size=(int(w), int(h)),
    )
