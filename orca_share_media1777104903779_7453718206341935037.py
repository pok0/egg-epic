import os
import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

# ─── Optional YOLO ────────────────────────────────────────────────────────────
# If `ultralytics` is installed AND model files exist in models/, we use YOLO.
# Otherwise we silently fall back to OpenCV detection — the app still works.
try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False

EGG_MODEL_PATH  = "models/egg_model.pt"
COIN_MODEL_PATH = "models/coin_model.pt"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thai Egg Sorter",
    page_icon="🥚",
    layout="centered",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main-header {
    background: linear-gradient(135deg, #f9f3e8 0%, #fde8c8 100%);
    border: 1px solid #f0c890;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 24px;
  }
  .main-header h1 { font-size: 18px; font-weight: 700; color: #5a3a1a; margin: 0; line-height: 1.4; }
  .result-card {
    background: #fff;
    border: 1px solid #dbe2e8;
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
  }
  .result-header {
    background: #f4f8fc;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    margin-bottom: 16px;
    border: 1px solid #dbe2e8;
  }
  .result-header h3 { color: #4A90E2; margin: 0; font-size: 20px; }
  .result-header p  { color: #888; margin: 4px 0 0; font-size: 13px; }
  .metric-row {
    display: flex; justify-content: space-between;
    padding: 10px 0; border-bottom: 1px solid #eee; font-size: 14px;
  }
  .metric-label { color: #555; font-weight: 500; }
  .metric-value { color: #4A90E2; font-weight: 600; }
  .section-title {
    font-size: 13px; font-weight: 700; color: #333;
    margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .detect-box {
    background: #f8fafc; border: 1px dashed #c0cfe0;
    border-radius: 10px; padding: 14px; margin-bottom: 12px;
    font-size: 13px; color: #555;
  }
  .coin-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 700; margin-left: 8px;
  }
  .coin-gold  { background: #fff8e1; color: #b8860b; border: 1px solid #f0c040; }
  .coin-silver{ background: #f0f4f8; color: #4a6080; border: 1px solid #c0cfe0; }
  .conf-high  { color: #2e7d32; font-weight: 600; }
  .conf-low   { color: #e65100; font-weight: 600; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px; font-weight: 600; font-size: 13px; padding: 8px 16px;
  }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🥚 Scale-Free Thai Chicken Egg Sorting:<br>
  Image Processing and Mathematical Model<br>
  for SMEs and Consumer Choice</h1>
</div>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
EGG_PARAMS = {
    0: dict(g=0.21, t=27.55, shell_thick=0.053, white_pct=0.712, yolk_pct=0.288,
            w=69.43, p=1.0647, label="Grade 0 (Jumbo ≥65 g)"),
    1: dict(g=0.21, t=26.90, shell_thick=0.050, white_pct=0.698, yolk_pct=0.302,
            w=67.07, p=1.0645, label="Grade 1 (Large 60–65 g)"),
    2: dict(g=0.23, t=26.59, shell_thick=0.050, white_pct=0.682, yolk_pct=0.318,
            w=61.02, p=1.0639, label="Grade 2 (Medium 55–60 g)"),
    3: dict(g=0.22, t=25.14, shell_thick=0.060, white_pct=0.684, yolk_pct=0.307,
            w=58.21, p=1.0636, label="Grade 3 (Small <55 g)"),
}

COIN_DIAMETERS_MM = {
    "10 Baht (26 mm)": 26.0,
    "5 Baht (24 mm)":  24.0,
    "1 Baht (20 mm)":  20.0,
}

EGG_SHELL_VOLUME_RATIO = 0.065
VOLUME_OL_AIR = 0.05

MM3_TO_ML = 1 / 1000.0   # 1 mm³ = 0.001 ml


# ─── Helpers ──────────────────────────────────────────────────────────────────
def ml(v: float) -> str:
    """Format a mm³ value as ml."""
    return f"{v * MM3_TO_ML:.4f} ml"

def auto_grade(a_mm: float) -> int:
    """Estimate Thai egg grade from half-length a (mm)."""
    length = a_mm * 2
    if length >= 65:   return 0
    elif length >= 60: return 1
    elif length >= 55: return 2
    else:              return 3


# ─── Core Math ────────────────────────────────────────────────────────────────
def compute_egg_metrics(a: float, b_max: float, size: int) -> dict:
    ep = EGG_PARAMS[size]
    g, t, p = ep["g"], ep["t"], ep["p"]
    white_pct, yolk_pct = ep["white_pct"], ep["yolk_pct"]
    shell_thick = ep["shell_thick"]

    b = (b_max * a) / (2 * a - t)
    k = -2 * (g + (g ** 3) / 2) / b

    # Volume
    V = (
        ((2 / 9) * math.pi * a * (b ** 2))
        * (
            (10 * ((1 / (9 - 6*k*a)) + (1 / (9 - 6*k*a))))
            + (8 * ((1 / (9 - 3*k*a)) + (1 / (9 + 3*k*a))))
            + 1
        )
    )

    def safe_sqrt(x):
        return math.sqrt(x) if x >= 0 else 0.0

    # Arc Length
    ArcLength = (math.pi / 18) * (
        (b * ((1 / math.sqrt(1 + k*a)) - (1 / math.sqrt(1 + k*a))))
        + 2 * (
            safe_sqrt(a**2 + ((b**2) * (7*k*a - 4*math.sqrt(3))**2) / (2*(2 - math.sqrt(3)*k*a))**3)
            + safe_sqrt(a**2 + ((b**2) * (7*k*a + 4*math.sqrt(3))**2) / (2*(2 + math.sqrt(3)*k*a))**3)
            + (a * math.sqrt(1 + ((b**2) * (k**2)) / 4))
        )
        + safe_sqrt(3*a**2 + ((b**2) * (5*k*a - 4)**2) / (2*(2 - k*a))**3)
        + safe_sqrt(3*a**2 + ((b**2) * (5*k*a + 4)**2) / (2*(2 + k*a))**3)
    )

    # Surface Area
    surface_area = (
        ((2 * math.pi * a * b) / 9)
        * (
            4 * math.sqrt((3/(3 - 2*k*a)) * (5/9 + (b**2*(13*k*a - 12)**2) / (12*(3 - 2*k*a)**3)))
            +   math.sqrt((3/(3 + 2*k*a)) * (5/9 + (b**2*(13*k*a + 12)**2) / (12*(3 + 2*k*a)**3)))
            + 2 * math.sqrt((3/(3 - k*a))  * (8/9 + (b**2*(10*k*a - b)**2)  / (12*(3 - k*a)**3)))
            +   math.sqrt((3/(3 + k*a))  * (8/9 + (b**2*(10*k*a + b)**2)  / (12*(3 + k*a)**3)))
            + (
                math.sqrt(1 + (a**2 * b**2 * k**2) / 4)
                - (2*a*b*k) / (1 - a**2*k**2)
            )
        )
    )

    surface_area_approx = (a / 2) * (
        (b / a) * (1/(1 + k*a) - 1/(1 - k*a))
        + 2 * safe_sqrt(1 + (b**2 * k**2) / 4)
    )

    shell_volume   = surface_area * EGG_SHELL_VOLUME_RATIO
    usable_volume  = V * (1 - VOLUME_OL_AIR)
    edible_volume  = usable_volume - shell_volume
    egg_white      = white_pct * edible_volume
    egg_yolk       = yolk_pct  * edible_volume
    albumin        = egg_white * ((0.59 * p) / 1.03)
    folic_acid     = edible_volume * (((0.47e-6) / 1.6) * p)
    iron           = edible_volume * (((0.0172e-3) / 7.8) * p)
    choline        = edible_volume * (((2.85e-3) / 1.09) * p)
    zinc           = edible_volume * (((0.011e-3) / 7.14) * p)
    vit_b12        = edible_volume * (((0.027e-6) / 0.52) * p)

    return dict(
        b=b, k=k, g=g, t=t, shell_thick=shell_thick, p=p,
        V=V, surface_area=surface_area, surface_area_approx=surface_area_approx,
        ArcLength=ArcLength, shell_volume=shell_volume,
        usable_volume=usable_volume, edible_volume=edible_volume,
        white_pct=white_pct, yolk_pct=yolk_pct,
        egg_white=egg_white, egg_yolk=egg_yolk,
        albumin=albumin, folic_acid=folic_acid,
        iron=iron, choline=choline, zinc=zinc, vit_b12=vit_b12,
    )


# ─── YOLO detection (custom-trained models) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_yolo_models():
    """Load both YOLO models once and cache them across reruns."""
    if not _ULTRALYTICS_AVAILABLE:
        return None, None, "ultralytics not installed"
    egg_m, coin_m, msg = None, None, ""
    try:
        if os.path.exists(EGG_MODEL_PATH):
            egg_m = YOLO(EGG_MODEL_PATH)
        else:
            msg += f"Egg model not found at {EGG_MODEL_PATH}. "
    except Exception as e:
        msg += f"Egg model load failed: {e}. "
    try:
        if os.path.exists(COIN_MODEL_PATH):
            coin_m = YOLO(COIN_MODEL_PATH)
        else:
            msg += f"Coin model not found at {COIN_MODEL_PATH}. "
    except Exception as e:
        msg += f"Coin model load failed: {e}. "
    return egg_m, coin_m, msg


def detect_egg_yolo(img_bgr: np.ndarray, egg_model):
    """
    YOLO egg detection + ellipse refinement.
    YOLO finds *where* the egg is; brown-blob segmentation inside the box
    produces a precise ellipse that follows the egg's actual outline (not
    the axis-aligned box, which over-estimates if the egg is tilted).
    Returns OpenCV ellipse tuple or None.
    """
    if egg_model is None:
        return None
    try:
        results = egg_model(img_bgr, verbose=False, conf=0.25)[0]
    except Exception:
        return None
    if results.boxes is None or len(results.boxes) == 0:
        return None

    # Pick highest-confidence detection
    confs = results.boxes.conf.cpu().numpy()
    best_idx = int(confs.argmax())
    x1, y1, x2, y2 = map(int, results.boxes.xyxy[best_idx].cpu().numpy())

    # ── Refine with ellipse fit inside the YOLO box ───────────────────────
    h, w = img_bgr.shape[:2]
    pad  = 12
    x1p  = max(0, x1 - pad)
    y1p  = max(0, y1 - pad)
    x2p  = min(w, x2 + pad)
    y2p  = min(h, y2 + pad)

    crop      = img_bgr[y1p:y2p, x1p:x2p]
    crop_mask = _brown_blob_mask(crop)
    cnts, _   = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        if len(largest) >= 5 and cv2.contourArea(largest) > 100:
            ellipse = cv2.fitEllipse(largest)
            (cx, cy), (MA, mi), angle = ellipse
            # Translate centre back to original image coords
            return ((cx + x1p, cy + y1p), (MA, mi), angle)

    # Fallback: derive ellipse straight from the YOLO bounding box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = x2 - x1, y2 - y1
    MA, mi = max(bw, bh), min(bw, bh)
    angle  = 0 if bh >= bw else 90       # tall box → vertical egg
    return ((cx, cy), (MA, mi), angle)


def detect_coin_yolo(img_bgr: np.ndarray, coin_model, egg_ellipse=None):
    """
    YOLO coin detection. Returns np.array([cx, cy, r], int) or None.
    If multiple coins are detected, the highest-confidence one whose centre
    falls outside the egg ellipse is chosen.
    """
    if coin_model is None:
        return None
    try:
        results = coin_model(img_bgr, verbose=False, conf=0.25)[0]
    except Exception:
        return None
    if results.boxes is None or len(results.boxes) == 0:
        return None

    h, w = img_bgr.shape[:2]
    egg_mask = _egg_center_mask(egg_ellipse, h, w)

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    valid = []
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if egg_mask[int(min(cy, h-1)), int(min(cx, w-1))] == 0:
            valid.append((box, conf))

    chosen = max(valid, key=lambda t: t[1])[0] if valid \
             else boxes[int(confs.argmax())]
    x1, y1, x2, y2 = chosen
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    # Use the *average* of width & height — the coin is round so they
    # should be ~equal; averaging cancels small box jitter.
    r = int(((x2 - x1) + (y2 - y1)) / 4.0)
    return np.array([cx, cy, r], dtype=int)


# ─── Image Processing ─────────────────────────────────────────────────────────
# Strategy
# ─────────
# EGG  → primary: brown HSV colour blob (Thai brown eggs are H 5-22, S 70-220)
#        fallback: grayscale CLAHE + Canny edge contours
# COIN → Hough circles with HARD size cap (≤11 % of image short-side radius)
#        This is the key fix: a ceiling light / lamp is 30-50 % of frame →
#        automatically excluded. A Thai coin at arm's length is 4-10 %.
#        After finding circles we also reject any circle whose centre sits
#        inside the detected egg blob.
# NOTE → True YOLO would give better accuracy but requires a custom-trained
#        model file (eggs and Thai coins are not in COCO). The colour-blob
#        approach here is the practical alternative for Streamlit Cloud.

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ── Egg detection ─────────────────────────────────────────────────────────────

def _brown_blob_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Isolate Thai brown egg pixels via HSV colour segmentation then clean with
    morphology.  Works in grayscale downstream so colour accuracy is forgiving.
    Hue 5-22  (brown/tan)
    Sat 70-230 (not washed-out, not pure grey)
    Val 50-210 (not pure black or pure white)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([5,  70,  50], np.uint8),
                       np.array([22, 230, 210], np.uint8))
    # Fill holes + remove small specks
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,  9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=2)
    return mask


def _gray_edge_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Grayscale-first fallback: CLAHE → bilateral → Canny → morphological close.
    Used when colour segmentation fails (e.g. white egg or unusual lighting).
    """
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    gcl   = clahe.apply(gray)
    bil   = cv2.bilateralFilter(gcl, 11, 80, 80)
    edges = cv2.Canny(bil, 15, 50)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    return edges


def _best_egg_contour(cnts, h: int, w: int):
    """Pick the most egg-like contour from a list. Returns (ellipse, score)."""
    best, best_score = None, 0.0
    for cnt in cnts:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        # Egg occupies roughly 2–40 % of image area
        if area < h * w * 0.02 or area > h * w * 0.50:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (MA, mi), _ = ellipse
        if mi < 1:
            continue
        ratio = MA / mi
        # Egg aspect ratio: 1.15–2.2
        if ratio < 1.15 or ratio > 2.2:
            continue
        # Solidity: egg contour should be fairly convex
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity   = area / hull_area if hull_area > 0 else 0
        if solidity < 0.75:
            continue
        # Penalise border-touching contours (likely background bleed)
        bm = 0.05
        border_pen = 0.25 if (cx < w*bm or cx > w*(1-bm) or
                               cy < h*bm or cy > h*(1-bm)) else 1.0
        # Sweet-spot bonus for typical egg shape
        shape_bonus = 1.4 if 1.25 <= ratio <= 1.85 else 1.0
        score = area * solidity * border_pen * shape_bonus
        if score > best_score:
            best_score, best = score, ellipse
    return best, best_score


def detect_egg_ellipse(img_bgr: np.ndarray):
    """
    Detect egg ellipse.  Returns OpenCV ellipse tuple or None.

    Pass 1 — brown colour blob (best for Thai brown eggs)
    Pass 2 — grayscale edge fallback (white eggs / unusual lighting)
    """
    h, w = img_bgr.shape[:2]

    # Pass 1: colour blob
    mask  = _brown_blob_mask(img_bgr)
    cnts1, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipse1, score1 = _best_egg_contour(cnts1, h, w)

    # Pass 2: grayscale edges (fallback)
    edges = _gray_edge_mask(img_bgr)
    cnts2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipse2, score2 = _best_egg_contour(cnts2, h, w)

    # Return whichever pass scored higher
    if score1 >= score2:
        return ellipse1
    return ellipse2


# ── Coin detection ────────────────────────────────────────────────────────────

def _egg_center_mask(ellipse, h: int, w: int) -> np.ndarray:
    """
    Return a binary mask filled inside the egg ellipse (inflated 10 %).
    Used to disqualify Hough circles whose centre lands inside the egg.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if ellipse is None:
        return mask
    (cx, cy), (MA, mi), angle = ellipse
    scaled = ((cx, cy), (MA * 1.1, mi * 1.1), angle)
    cv2.ellipse(mask, scaled, 255, -1)
    return mask


def _hough_coin_circles(gray: np.ndarray, h: int, w: int):
    """
    Hough circle detection with a HARD maximum radius cap.

    Cap = 11 % of shorter image dimension.
    A Thai coin at normal photo distance is 4–10 %.
    A lamp / light fixture is typically 25–50 % → safely excluded.
    """
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_cl = clahe.apply(gray)

    min_r = int(min(h, w) * 0.015)   # ~smallest detectable coin
    max_r = int(min(h, w) * 0.11)    # HARD CAP — kills background circles

    all_circles = []
    variants = [
        (cv2.GaussianBlur(gray,    (9, 9), 2), 1.2, 60, 28),
        (cv2.GaussianBlur(gray,    (7, 7), 1), 1.0, 70, 35),
        (cv2.GaussianBlur(gray_cl, (9, 9), 2), 1.2, 55, 25),
        (cv2.GaussianBlur(gray_cl, (7, 7), 1), 1.5, 50, 22),
    ]
    for src, dp, p1, p2 in variants:
        circles = cv2.HoughCircles(
            src, cv2.HOUGH_GRADIENT, dp=dp,
            minDist=min(h, w) * 0.05,
            param1=p1, param2=p2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            all_circles.extend(np.round(circles[0]).astype(int).tolist())
    return all_circles


def _classify_coin_color(img_bgr: np.ndarray, cx: int, cy: int, r: int) -> tuple:
    """
    Identify Thai coin by HSV analysis of the outer ring (65-100 % of radius).
    10 Baht: bi-metallic → gold/brass outer ring (H 10-40, S > 55, V > 80)
     5 Baht: silver nickel → low S, high V
     1 Baht: smaller silver nickel → same as 5 Baht but smaller
    """
    h_img, w_img = img_bgr.shape[:2]
    m_out = np.zeros((h_img, w_img), np.uint8)
    m_in  = np.zeros((h_img, w_img), np.uint8)
    cv2.circle(m_out, (cx, cy), r,           255, -1)
    cv2.circle(m_in,  (cx, cy), int(r*0.65), 255, -1)
    ring  = cv2.subtract(m_out, m_in)

    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    px    = hsv[ring > 0]
    if len(px) == 0:
        return "5 Baht (24 mm)", 0.4, "silver"

    H, S, V = px[:, 0], px[:, 1], px[:, 2]
    n       = len(px)
    gold_r  = ((H >= 10) & (H <= 40) & (S > 55) & (V > 80)).sum() / n

    if gold_r > 0.10:
        return "10 Baht (26 mm)", round(min(gold_r * 5, 1.0), 2), "gold"

    rel = r / min(h_img, w_img)
    if rel > 0.06:
        return "5 Baht (24 mm)", 0.55, "silver"
    return "1 Baht (20 mm)", 0.55, "silver"


def detect_coin(img_bgr: np.ndarray, egg_ellipse=None):
    """
    Detect + classify coin.
    egg_ellipse: pass the detected egg so we can ignore circles inside it.
    Returns (circle_arr, type_str, conf, colour_tag) — circle_arr may be None.
    """
    h, w   = img_bgr.shape[:2]
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    egg_mask = _egg_center_mask(egg_ellipse, h, w)

    circles  = _hough_coin_circles(gray, h, w)
    if not circles:
        return None, None, 0.0, "silver"

    # Filter out circles whose centre is inside the egg
    valid = [c for c in circles if egg_mask[min(c[1], h-1), min(c[0], w-1)] == 0]
    if not valid:
        valid = circles   # edge case: keep all if filtering wiped everything

    best = max(valid, key=lambda c: c[2])
    cx, cy, r = int(best[0]), int(best[1]), int(best[2])
    coin_type, conf, ctag = _classify_coin_color(img_bgr, cx, cy, r)
    return np.array([cx, cy, r], dtype=int), coin_type, conf, ctag


def detect_coin_manual(img_bgr: np.ndarray, egg_ellipse=None):
    """Detect largest valid circle only (user supplies coin type)."""
    h, w   = img_bgr.shape[:2]
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    egg_mask = _egg_center_mask(egg_ellipse, h, w)

    circles = _hough_coin_circles(gray, h, w)
    if not circles:
        return None
    valid = [c for c in circles if egg_mask[min(c[1], h-1), min(c[0], w-1)] == 0]
    if not valid:
        valid = circles
    return np.array(max(valid, key=lambda c: c[2]), dtype=int)


def annotate_image(img_bgr: np.ndarray, ellipse, coin, coin_label="") -> np.ndarray:
    out = img_bgr.copy()
    if ellipse is not None:
        cv2.ellipse(out, ellipse, (0, 200, 60), 3)
        cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
        cv2.circle(out, (cx, cy), 5, (0, 200, 60), -1)
        cv2.putText(out, "EGG", (cx - 20, cy - int(ellipse[1][0] / 2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 60), 2)
    if coin is not None:
        cx, cy, r = int(coin[0]), int(coin[1]), int(coin[2])
        cv2.circle(out, (cx, cy), r, (255, 140, 0), 3)
        lbl = coin_label if coin_label else "COIN"
        cv2.putText(out, lbl, (cx - 30, cy - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 140, 0), 2)
    return out


# ─── Result Display ───────────────────────────────────────────────────────────
def show_results(r: dict, size: int, a_input: float, b_max_input: float):
    label = EGG_PARAMS[size]["label"]
    length_disp = a_input   * 2
    width_disp  = b_max_input * 2

    st.markdown(f"""
    <div class="result-card">
      <div class="result-header">
        <h3>🏆 {label}</h3>
        <p>Length = {length_disp:.2f} mm &nbsp;|&nbsp; Width = {width_disp:.2f} mm
        &nbsp;|&nbsp; (a = {a_input:.2f}, b_max = {b_max_input:.2f})</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Volume",         ml(r['V']))
        st.metric("Surface Area",   f"{r['surface_area']:.4f} mm²")
        st.metric("Arc Length",     f"{r['ArcLength']:.4f} mm")
        st.metric("Shell Volume",   ml(r['shell_volume']))
    with col2:
        st.metric("Usable Volume",  ml(r['usable_volume']))
        st.metric("Edible Volume",  ml(r['edible_volume']))
        st.metric("b (adjusted)",   f"{r['b']:.4f} mm")
        st.metric("k",              f"{r['k']:.6f}")

    with st.expander("🥚 Egg White & Yolk Breakdown"):
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Egg White",     ml(r['egg_white']))
            st.metric("White Fraction", f"{r['white_pct']*100:.1f}%")
        with c2:
            st.metric("Egg Yolk",      ml(r['egg_yolk']))
            st.metric("Yolk Fraction", f"{r['yolk_pct']*100:.1f}%")

    with st.expander("🧬 Nutritional Composition (volume-based)"):
        cols = st.columns(2)
        nutrients = [
            ("Albumin",     r["albumin"]),
            ("Folic Acid",  r["folic_acid"]),
            ("Iron",        r["iron"]),
            ("Choline",     r["choline"]),
            ("Zinc",        r["zinc"]),
            ("Vitamin B12", r["vit_b12"]),
        ]
        for i, (lbl, val) in enumerate(nutrients):
            with cols[i % 2]:
                st.metric(lbl, ml(val))

    with st.expander("📐 Model Parameters"):
        st.code(
            f"Size:              {size} — {EGG_PARAMS[size]['label']}\n"
            f"g:                 {r['g']}\n"
            f"t:                 {r['t']}\n"
            f"shell thickness:   {r['shell_thick']} mm\n"
            f"density (p):       {r['p']}\n"
            f"b (adjusted):      {r['b']:.6f} mm\n"
            f"k:                 {r['k']:.6f}\n"
            f"Surface Area (SA): {r['surface_area']:.6f} mm²\n"
            f"SA approx:         {r['surface_area_approx']:.6f} mm²",
            language="text",
        )


# ─── Tabs ─────────────────────────────────────────────────────────────────────
# ─── Tabs ─────────────────────────────────────────────────────────────────────
# Load YOLO models once (cached). Status banner tells the user which mode is active.
egg_yolo, coin_yolo, yolo_msg = load_yolo_models()
USE_YOLO_EGG  = egg_yolo  is not None
USE_YOLO_COIN = coin_yolo is not None

if USE_YOLO_EGG and USE_YOLO_COIN:
    st.success("🤖 **YOLO mode active** — using custom-trained AI models for both egg and coin.")
elif USE_YOLO_EGG or USE_YOLO_COIN:
    st.info(f"🤖 Partial YOLO: egg={'✅' if USE_YOLO_EGG else '❌'} · "
            f"coin={'✅' if USE_YOLO_COIN else '❌'} (missing parts use OpenCV).")
else:
    st.info("🔧 OpenCV mode (no YOLO models loaded). " +
            (f"Note: {yolo_msg}" if yolo_msg else ""))

tab1, tab2, tab3 = st.tabs(["📸 Photo + Coin", "📸 Photo + Grade", "⌨️ Manual Input"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PHOTO + COIN  (no grade needed; auto-coin detection option)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("**Upload a photo of the egg with a Thai coin beside it for scale.**")

    coin_mode = st.radio(
        "Coin identification",
        ["🔍 Auto-detect coin type", "✏️ Manual select"],
        horizontal=True,
        key="coin_mode",
    )

    manual_coin_type = None
    if coin_mode == "✏️ Manual select":
        manual_coin_type = st.selectbox(
            "Select coin type",
            list(COIN_DIAMETERS_MM.keys()),
            key="manual_coin_sel",
        )

    uploaded_coin = st.file_uploader(
        "Upload egg + coin image", type=["jpg", "jpeg", "png"], key="up_coin"
    )

    if uploaded_coin:
        pil_img = Image.open(uploaded_coin)
        img_bgr = pil_to_cv(pil_img)

        with st.spinner("🔍 Detecting egg and coin…"):
            # Egg
            if USE_YOLO_EGG:
                ellipse = detect_egg_yolo(img_bgr, egg_yolo)
            else:
                ellipse = detect_egg_ellipse(img_bgr)

            # Coin
            if coin_mode == "🔍 Auto-detect coin type":
                if USE_YOLO_COIN:
                    coin_arr      = detect_coin_yolo(img_bgr, coin_yolo, egg_ellipse=ellipse)
                    if coin_arr is not None:
                        cx_, cy_, r_ = int(coin_arr[0]), int(coin_arr[1]), int(coin_arr[2])
                        detected_type, conf, ctag = _classify_coin_color(img_bgr, cx_, cy_, r_)
                    else:
                        detected_type, conf, ctag = None, 0.0, "silver"
                else:
                    coin_arr, detected_type, conf, ctag = detect_coin(img_bgr, egg_ellipse=ellipse)
            else:
                if USE_YOLO_COIN:
                    coin_arr = detect_coin_yolo(img_bgr, coin_yolo, egg_ellipse=ellipse)
                else:
                    coin_arr = detect_coin_manual(img_bgr, egg_ellipse=ellipse)
                detected_type = manual_coin_type
                conf, ctag    = 1.0, "silver"

        # Annotation label shown on image
        short_label = detected_type.split(" (")[0] if detected_type else "COIN"
        annotated = annotate_image(img_bgr, ellipse, coin_arr, short_label)
        st.image(cv_to_pil(annotated), caption="Detection result", use_container_width=True)

        if ellipse is None:
            st.warning("⚠️ Could not detect egg contour. Tips: use a plain/contrasting background, avoid harsh shadows, and ensure the whole egg is in frame.")
        elif coin_arr is None:
            st.warning("⚠️ Coin not detected. Place the coin clearly on a flat surface beside the egg with no overlap.")
        else:
            coin_real_mm = COIN_DIAMETERS_MM[detected_type]
            coin_px      = coin_arr[2] * 2
            mm_per_px    = coin_real_mm / coin_px

            major_ax_px = max(ellipse[1])
            minor_ax_px = min(ellipse[1])
            a_val       = (major_ax_px * mm_per_px) / 2
            b_max_val   = (minor_ax_px * mm_per_px) / 2

            # Auto-determine grade from egg size
            auto_size = auto_grade(a_val)

            # Coin confidence line
            if coin_mode == "🔍 Auto-detect coin type":
                badge_cls  = "coin-gold" if ctag == "gold" else "coin-silver"
                conf_cls   = "conf-high" if conf >= 0.65 else "conf-low"
                conf_pct   = f"{conf*100:.0f}%"
                coin_conf_html = (
                    f'<span class="coin-badge {badge_cls}">{detected_type}</span>'
                    f' &nbsp; <span class="{conf_cls}">confidence {conf_pct}</span>'
                )
            else:
                coin_conf_html = f'<b>Coin:</b> {detected_type} (manual)'

            st.markdown(f"""
            <div class="detect-box">
              🪙 {coin_conf_html}<br>
              &nbsp;&nbsp;&nbsp;Coin diameter: {coin_px:.1f} px → {coin_real_mm} mm
              &nbsp;|&nbsp; Scale: {mm_per_px:.4f} mm/px<br>
              🥚 <b>Egg:</b> long axis {major_ax_px:.1f} px ({major_ax_px*mm_per_px:.2f} mm),
              short axis {minor_ax_px:.1f} px ({minor_ax_px*mm_per_px:.2f} mm)<br>
              📐 <b>a</b> = {a_val:.2f} mm &nbsp;|&nbsp; <b>b_max</b> = {b_max_val:.2f} mm<br>
              🏷️ <b>Auto-graded:</b> {EGG_PARAMS[auto_size]['label']}
            </div>
            """, unsafe_allow_html=True)

            if coin_mode == "🔍 Auto-detect coin type" and conf < 0.65:
                st.info(
                    f"ℹ️ Low confidence on coin type. "
                    f"Auto-selected **{detected_type}** — switch to Manual select if incorrect."
                )

            if st.button("▶ Run Precision Analysis", key="run_coin"):
                try:
                    results = compute_egg_metrics(a_val, b_max_val, auto_size)
                    show_results(results, auto_size, a_val, b_max_val)
                except Exception as e:
                    st.error(f"Calculation error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PHOTO + GRADE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("**Upload a photo of the egg. Select the known Thai grade to scale dimensions.**")

    GRADE_BASELINE_LENGTH_MM = {0: 65.0, 1: 61.0, 2: 57.5, 3: 54.0}

    known_grade    = st.selectbox("Known egg grade", list(EGG_PARAMS.keys()),
                                  format_func=lambda x: EGG_PARAMS[x]["label"], key="known_grade")
    uploaded_grade = st.file_uploader("Upload egg image (no coin needed)",
                                       type=["jpg", "jpeg", "png"], key="up_grade")

    if uploaded_grade:
        pil_img2 = Image.open(uploaded_grade)
        img_bgr2 = pil_to_cv(pil_img2)

        with st.spinner("🔍 Detecting egg shape…"):
            if USE_YOLO_EGG:
                ellipse2 = detect_egg_yolo(img_bgr2, egg_yolo)
            else:
                ellipse2 = detect_egg_ellipse(img_bgr2)
            annotated2 = annotate_image(img_bgr2, ellipse2, None)

        st.image(cv_to_pil(annotated2), caption="Detection result", use_container_width=True)

        if ellipse2 is None:
            st.warning("⚠️ Could not detect egg. Try better lighting or a plain background.")
        else:
            major_px = max(ellipse2[1])
            minor_px = min(ellipse2[1])
            ratio    = minor_px / major_px

            baseline_len_mm = GRADE_BASELINE_LENGTH_MM[known_grade]
            length_mm = baseline_len_mm
            width_mm  = baseline_len_mm * ratio
            a_val2    = length_mm / 2
            b_max_val2 = width_mm / 2

            st.markdown(f"""
            <div class="detect-box">
              🥚 <b>Detected pixel ratio</b> (w/l): {ratio:.4f}<br>
              📐 Scaled to Grade {known_grade} baseline:
              length = {length_mm:.2f} mm, width = {width_mm:.2f} mm<br>
              <b>a</b> = {a_val2:.2f} mm &nbsp;|&nbsp; <b>b_max</b> = {b_max_val2:.2f} mm
            </div>
            """, unsafe_allow_html=True)

            if st.button("▶ Run Precision Analysis", key="run_grade"):
                try:
                    results2 = compute_egg_metrics(a_val2, b_max_val2, known_grade)
                    show_results(results2, known_grade, a_val2, b_max_val2)
                except Exception as e:
                    st.error(f"Calculation error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MANUAL INPUT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("**Enter egg dimensions directly (no image required).**")

    col_a, col_b = st.columns(2)
    with col_a:
        length_manual = st.number_input(
            "Length — tip to tip (mm)",
            min_value=2.0, max_value=200.0, value=60.0, step=0.1, key="len_man"
        )
    with col_b:
        width_manual = st.number_input(
            "Width — widest diameter (mm)",
            min_value=2.0, max_value=160.0, value=44.0, step=0.1, key="wid_man"
        )

    size_manual = st.selectbox("Egg grade (Thai standard)", list(EGG_PARAMS.keys()),
                               format_func=lambda x: EGG_PARAMS[x]["label"], key="size_man")
    st.caption("💡 Measure tip-to-tip for **Length** and the widest middle point for **Width**."
               " Values are halved internally (a = Length÷2, b_max = Width÷2).")

    if st.button("▶ Run Precision Analysis", key="run_manual"):
        try:
            a_man    = float(length_manual) / 2.0
            bmax_man = float(width_manual)  / 2.0
            results3 = compute_egg_metrics(a_man, bmax_man, size_manual)
            show_results(results3, size_manual, a_man, bmax_man)
        except Exception as e:
            st.error(f"Calculation error: {e}")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Mathematical model adapted from Narushin-type ovoid equations. "
    "Detection: custom-trained YOLOv8 models for egg & coin (when available), "
    "with brown HSV ellipse refinement for precise measurement. "
    "Falls back to OpenCV (colour segmentation + size-bounded Hough) if models are absent. "
    "Thai egg grades 0–3 (TFDA standard). All volumes in ml (1 ml = 1 000 mm³)."
)
