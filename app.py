"""
Stage 2: 제루미 실제 분석 로직(calibration, skin_extraction, recommend)을
그대로 옮겨온 진짜 백엔드 API.

Stage 1에서 확인된 것들이 전부 반영돼 있어요:
- Vercel "Large Functions" 켜짐 (용량 500MB -> 5GB)
- mediapipe용 opencv는 화면 필요 없는 버전(opencv-contrib-python-headless)
- scikit-image 대신 순수 numpy로 직접 구현한 색상 변환 (src/color_match.py)

데이터(파운데이션 색상 목록)는 지금까지 쓰던 방식 그대로, GitHub 저장소의
data/foundation_db.json 파일을 읽고 씁니다 (환경변수 GITHUB_TOKEN,
GITHUB_REPO 설정 필요 -- 아래 read_shades() 참고). 환경변수가 아직
설정되지 않았으면 이 프로젝트에 같이 들어있는 로컬 복사본으로 대신
동작해요 (테스트용 fallback).
"""
import json
import os
import traceback
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import cv2
from fastapi import Body, FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from src.calibration import apply_correction, calibrate_from_image, capture_card_reference
from src.color_match import rgb_to_lab
from src.recommend import recommend_foundation
from src import github_storage
from src import reference_colors

app = FastAPI()

# 프론트엔드(Next.js, 다른 Vercel 프로젝트)에서 이 API를 호출할 수 있도록 허용.
# 지금은 별도 프로젝트로 나눠서 배포하기 때문에 이 설정이 없으면 브라우저가
# 보안상 요청을 막아버려요 (CORS 오류).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCAL_DB_PATH = Path(__file__).parent / "data" / "foundation_db.json"


def load_shades():
    """GitHub 저장소에서 읽기를 먼저 시도하고, 환경변수가 없거나 실패하면
    이 프로젝트에 같이 들어있는 로컬 복사본으로 대체해요."""
    try:
        shades, _sha = github_storage.read_shades()
        if shades:
            return shades, "github"
    except Exception:
        pass
    with open(LOCAL_DB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("shades", []), "local_fallback"


def _to_jsonable(obj):
    """numpy 배열/스칼라를 JSON으로 바로 못 바꾸는 문제를 해결하기 위한
    재귀 변환 헬퍼."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


@app.get("/")
def health():
    return {"status": "ok", "message": "jerumi 실제 분석 API가 정상적으로 떠 있어요"}


@app.get("/api/shades")
def api_shades():
    shades, source = load_shades()
    enriched = [{**s, "verified": s.get("verified", True)} for s in shades]
    n_verified = sum(1 for s in enriched if s["verified"])
    return {
        "source": source,
        "total": len(enriched),
        "n_verified": n_verified,
        "n_unverified": len(enriched) - n_verified,
        "shades": enriched,
    }


# ---------------------------------------------------------------------------
# 관리자 전용 -- 색상 DB 추가/삭제, 카드 기준값 재설정.
# 원래 Streamlit 버전의 pages/1_관리자.py 를 API 형태로 옮긴 거예요. 화면(UI)은
# 프론트엔드(Next.js) /admin 페이지 쪽에 있고, 여기는 그 화면이 호출하는
# 실제 처리 로직이에요.
#
# 비밀번호는 Vercel 환경변수 ADMIN_PASSWORD 에 설정해두면, 모든 관리자
# 요청에 담겨오는 X-Admin-Password 헤더 값과 비교해서 확인해요.
# ---------------------------------------------------------------------------


def _robust_average_rgb(rgb_pixels: np.ndarray) -> np.ndarray:
    """사진(또는 크롭한 영역) 안의 픽셀들에서 하이라이트(반사광으로 인한 밝은
    부분)와 그림자(어두운 부분)를 자동으로 제외하고 평균 색상을 구해요.

    방법: 각 픽셀의 밝기(luminance)를 계산해서, 밝기 기준 하위 15% / 상위
    15%에 해당하는 픽셀(=그림자, 하이라이트일 가능성이 높은 픽셀)은 버리고
    나머지(중간 밝기, 즉 대부분 고르게 조명된 '진짜' 색상 부분)만으로
    평균을 내요. 남는 픽셀이 너무 적으면(사진이 아주 작거나 단색에 가까우면)
    안전하게 전체 픽셀 평균으로 대신해요."""
    rgb_pixels = rgb_pixels.reshape(-1, 3).astype(np.float64)
    if len(rgb_pixels) == 0:
        return np.zeros(3)
    lum = rgb_pixels[:, 0] * 0.299 + rgb_pixels[:, 1] * 0.587 + rgb_pixels[:, 2] * 0.114
    lo, hi = np.percentile(lum, [15, 85])
    mask = (lum >= lo) & (lum <= hi)
    if mask.sum() < max(10, len(rgb_pixels) * 0.05):
        return np.mean(rgb_pixels, axis=0)
    return np.mean(rgb_pixels[mask], axis=0)


def _require_admin(x_admin_password: Optional[str]):
    expected = os.environ.get("ADMIN_PASSWORD")
    if not expected:
        raise HTTPException(status_code=500, detail="서버에 ADMIN_PASSWORD 환경변수가 설정되어 있지 않아요")
    if not x_admin_password or x_admin_password != expected:
        raise HTTPException(status_code=401, detail="비밀번호가 틀렸어요")


@app.post("/api/admin/login")
def admin_login(x_admin_password: Optional[str] = Header(None)):
    _require_admin(x_admin_password)
    return {"ok": True}


@app.post("/api/admin/shades")
async def admin_add_shade(
    brand: str = Form(...),
    name: str = Form(...),
    verified: bool = Form(...),
    use_card: bool = Form(True),
    crops: str = Form(...),
    files: List[UploadFile] = File(...),
    x_admin_password: Optional[str] = Header(None),
):
    """사진(최대 5장)으로 색상을 측정해서 새 파운데이션 색상을 저장해요.

    use_card=True 면 사진마다 색상카드를 찾아 카메라/조명 보정을 거치고,
    False 면 보정 없이 그대로 써요. 여러 장이면 평균을 내요 (원래 Streamlit
    버전과 동일한 방식).

    crops의 각 항목은 {x,y,w,h} 또는 null이에요. null이면(주로 "이미지에서
    색상만 추출" 방식처럼 이미 원하는 부분만 잘라서 올린 사진일 때) 크롭
    없이 사진 전체를 그대로 사용해요. 어느 쪽이든 실제 평균 색상은
    _robust_average_rgb()가 하이라이트(반사광)/그림자에 해당하는 픽셀을
    자동으로 제외하고 계산해요."""
    _require_admin(x_admin_password)

    try:
        crop_boxes = json.loads(crops)
    except Exception:
        return JSONResponse(status_code=400, content={"success": False, "message": "crops 형식이 올바르지 않아요"})

    if not isinstance(crop_boxes, list) or len(crop_boxes) != len(files):
        return JSONResponse(status_code=400, content={"success": False, "message": "사진 개수와 선택 영역 개수가 달라요"})

    labs = []
    photo_debug = []
    for f, box in zip(files, crop_boxes):
        contents = await f.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            photo_debug.append({"file": f.filename, "error": "이미지를 읽지 못했어요"})
            continue

        h_img, w_img = bgr.shape[:2]
        if box is None:
            # 크롭 지정 없음 -> 사진 전체를 그대로 사용해요.
            crop_bgr = bgr
        else:
            x = max(0, min(int(box.get("x", 0)), w_img - 1))
            y = max(0, min(int(box.get("y", 0)), h_img - 1))
            w = max(1, min(int(box.get("w", 1)), w_img - x))
            h = max(1, min(int(box.get("h", 1)), h_img - y))
            crop_bgr = bgr[y : y + h, x : x + w]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        avg_rgb = _robust_average_rgb(crop_rgb)

        if use_card:
            calib = calibrate_from_image(bgr)
            if not calib.success:
                photo_debug.append({"file": f.filename, "error": f"색상카드 인식 실패: {calib.message}"})
                continue
            corrected_rgb = apply_correction(avg_rgb, calib.correction_matrix)
            lab = rgb_to_lab(corrected_rgb)
            photo_debug.append(
                {
                    "file": f.filename,
                    "lab": [round(float(v), 2) for v in lab],
                    "calib_error": round(float(calib.mean_delta_e), 2),
                }
            )
        else:
            lab = rgb_to_lab(avg_rgb)
            photo_debug.append({"file": f.filename, "lab": [round(float(v), 2) for v in lab]})

        labs.append(lab)

    if not labs:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "사용할 수 있는 사진이 없어요", "photos": photo_debug},
        )

    mean_lab = np.mean(labs, axis=0)
    new_shade = {
        "id": f"{brand}-{name}-{uuid.uuid4().hex[:6]}".lower().replace(" ", "-"),
        "brand": brand,
        "name": name,
        "L": round(float(mean_lab[0]), 3),
        "a": round(float(mean_lab[1]), 3),
        "b": round(float(mean_lab[2]), 3),
        "verified": verified,
    }
    try:
        github_storage.add_shade(new_shade)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"저장 실패: {e}"})

    return {"success": True, "message": "저장 완료", "shade": new_shade, "photos": photo_debug}


@app.delete("/api/admin/shades/{shade_id}")
def admin_delete_shade(shade_id: str, x_admin_password: Optional[str] = Header(None)):
    _require_admin(x_admin_password)
    try:
        github_storage.delete_shade(shade_id)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"삭제 실패: {e}"})
    return {"success": True}


@app.get("/api/admin/card-reference")
def admin_get_card_reference(x_admin_password: Optional[str] = Header(None)):
    _require_admin(x_admin_password)
    try:
        data, _sha = github_storage.read_card_reference()
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})
    return {"success": True, "patches": (data or {}).get("patches") if data else None}


@app.post("/api/admin/card-reference/preview")
async def admin_card_reference_preview(
    file: UploadFile = File(...),
    x_admin_password: Optional[str] = Header(None),
):
    """카드만 깨끗하게 찍은 사진에서 24개 패치 색을 추출해서 미리보기로
    보여줘요. 아직 저장은 안 해요 (확인 후 /save 를 따로 호출)."""
    _require_admin(x_admin_password)
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse(status_code=400, content={"success": False, "message": "이미지를 읽지 못했어요"})
    ok, msg, patches = capture_card_reference(bgr)
    if not ok:
        return JSONResponse(status_code=400, content={"success": False, "message": msg})
    patches_json = {k: [round(float(c), 1) for c in v] for k, v in patches.items()}
    return {"success": True, "message": msg, "patches": patches_json}


@app.post("/api/admin/card-reference/save")
def admin_card_reference_save(
    payload: dict = Body(...),
    x_admin_password: Optional[str] = Header(None),
):
    _require_admin(x_admin_password)
    try:
        github_storage.write_card_reference(payload)
        reference_colors.clear_reference_cache()
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})
    return {"success": True}


@app.delete("/api/admin/card-reference")
def admin_card_reference_reset(x_admin_password: Optional[str] = Header(None)):
    _require_admin(x_admin_password)
    try:
        github_storage.delete_card_reference()
        reference_colors.clear_reference_cache()
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})
    return {"success": True}


@app.post("/api/recommend")
async def api_recommend(
    file: UploadFile = File(...),
    has_card: bool = Form(True),
    use_verified_only: bool = Form(False),
    top_n: int = Form(3),
):
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        bgr_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr_img is None:
            return JSONResponse(status_code=400, content={"success": False, "message": "이미지 파일을 읽지 못했어요"})

        shades, source = load_shades()
        shade_pool = [s for s in shades if s.get("verified", True)] if use_verified_only else shades
        if not shade_pool:
            return JSONResponse(status_code=400, content={"success": False, "message": "선택한 범위에 데이터가 없어요"})

        result = recommend_foundation(bgr_img, shade_pool, top_n=top_n, has_card=has_card)

        return _to_jsonable(
            {
                "success": result.success,
                "message": result.message,
                "corrected_lab": result.corrected_lab,
                "matches": result.matches,
                "debug": result.debug,
                "data_source": source,
                "shade_pool_size": len(shade_pool),
            }
        )
    except Exception as e:  # pragma: no cover - surfaced to the caller for debugging
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"서버 오류: {e}",
                "traceback": traceback.format_exc(),
            },
        )


# ---------------------------------------------------------------------------
# 간단한 테스트 페이지 -- 코딩 없이 브라우저에서 바로 사진 업로드해서
# /api/recommend 결과를 확인해볼 수 있어요. (실제 서비스용 화면이 아니라
# Stage 2 백엔드 로직 자체가 잘 동작하는지 확인하기 위한 용도예요.)
# ---------------------------------------------------------------------------
TEST_PAGE_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>jerumi API 테스트</title>
<style>
  body { font-family: sans-serif; max-width: 480px; margin: 40px auto; padding: 0 16px; }
  label { display: block; margin-top: 12px; font-weight: bold; }
  button { margin-top: 16px; padding: 10px 20px; font-size: 16px; }
  pre { background: #f5f5f5; padding: 12px; border-radius: 8px; white-space: pre-wrap; word-break: break-all; }
</style>
</head>
<body>
  <h2>jerumi API 테스트 페이지</h2>
  <form id="f">
    <label>사진 (카드+얼굴 또는 얼굴만)</label>
    <input type="file" name="file" accept="image/*" required>

    <label><input type="checkbox" name="has_card" checked> 색상카드 있음</label>
    <label><input type="checkbox" name="use_verified_only"> 정확측정 데이터만 사용</label>

    <button type="submit">추천 받기</button>
  </form>
  <p id="status"></p>
  <pre id="result"></pre>

<script>
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const fd = new FormData();
  fd.append('file', form.file.files[0]);
  fd.append('has_card', form.has_card.checked);
  fd.append('use_verified_only', form.use_verified_only.checked);
  fd.append('top_n', '3');
  document.getElementById('status').textContent = '분석 중... (몇 초 걸릴 수 있어요)';
  document.getElementById('result').textContent = '';
  try {
    const res = await fetch('/api/recommend', { method: 'POST', body: fd });
    const json = await res.json();
    document.getElementById('status').textContent = res.ok ? '완료' : '오류 (아래 내용 확인)';
    document.getElementById('result').textContent = JSON.stringify(json, null, 2);
  } catch (err) {
    document.getElementById('status').textContent = '요청 실패';
    document.getElementById('result').textContent = String(err);
  }
});
</script>
</body>
</html>
"""


@app.get("/test", response_class=HTMLResponse)
def test_page():
    return TEST_PAGE_HTML
