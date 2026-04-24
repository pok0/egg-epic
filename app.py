import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

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


# ─── Improved Image Processing ────────────────────────────────────────────────
def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _multi_edge(gray: np.ndarray) -> np.ndarray:
    """
    Combine three edge maps for robust detection on varied backgrounds.
    1. CLAHE-enhanced + Gaussian → Canny (low thresholds, catches faint edges)
    2. Bilateral filter → Canny  (preserves sharp edges, less noise)
    3. Otsu threshold → morphological gradient (handles uniform backgrounds)
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_cl = clahe.apply(gray)

    # (1)
    b1 = cv2.GaussianBlur(gray_cl, (7, 7), 0)
    e1 = cv2.Canny(b1, 15, 50)

    # (2)
    b2 = cv2.bilateralFilter(gray, 11, 75, 75)
    e2 = cv2.Canny(b2, 25, 80)

    # (3) morphological gradient = dilation - erosion → ring of edges
    kern3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil   = cv2.dilate(gray_cl, kern3)
    ero   = cv2.erode(gray_cl, kern3)
    grad  = cv2.subtract(dil, ero)
    _, e3 = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_or(cv2.bitwise_or(e1, e2), e3)

    # Close gaps so contours are closed loops
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k, iterations=2)
    combined = cv2.dilate(combined,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                          iterations=1)
    return combined


def detect_egg_ellipse(img_bgr: np.ndarray):
    """
    Robust egg-ellipse detection.
    Returns OpenCV ellipse tuple ((cx,cy),(MA,mi),angle) or None.
    """
    h, w = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = _multi_edge(gray)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_score = None, 0
    for cnt in contours:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if area < h * w * 0.005 or area > h * w * 0.85:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (MA, mi), _ = ellipse
        if mi < 1:
            continue
        ratio = MA / mi
        # Egg aspect ratios roughly 1.1–2.8
        if ratio < 1.05 or ratio > 3.0:
            continue
        # Penalise contours hugging the image border (likely frame artefacts)
        border_margin = 0.05
        if (cx < w*border_margin or cx > w*(1-border_margin) or
                cy < h*border_margin or cy > h*(1-border_margin)):
            area *= 0.3
        score = area
        if score > best_score:
            best_score = score
            best = ellipse

    return best


def _hough_circles_multi(gray: np.ndarray, h: int, w: int):
    """
    Try several Hough parameter sets and return all detected circles merged.
    """
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    # Also try with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blur_cl = cv2.GaussianBlur(clahe.apply(gray), (9, 9), 2)

    all_circles = []
    param_sets = [
        # (image, dp, p1, p2, minR_frac, maxR_frac)
        (blur,    1.2, 60, 30, 0.025, 0.28),
        (blur,    1.5, 50, 25, 0.020, 0.32),
        (blur,    1.0, 80, 40, 0.030, 0.25),
        (blur_cl, 1.2, 55, 28, 0.025, 0.28),
        (blur_cl, 1.5, 45, 22, 0.020, 0.32),
    ]
    for img_src, dp, p1, p2, mnf, mxf in param_sets:
        circles = cv2.HoughCircles(
            img_src, cv2.HOUGH_GRADIENT, dp=dp,
            minDist=min(h, w) * 0.07,
            param1=p1, param2=p2,
            minRadius=int(min(h, w) * mnf),
            maxRadius=int(min(h, w) * mxf),
        )
        if circles is not None:
            all_circles.extend(np.round(circles[0]).astype(int).tolist())

    return all_circles


def _classify_coin_by_color(img_bgr: np.ndarray, cx: int, cy: int, r: int) -> tuple:
    """
    Identify Thai coin denomination by analysing HSV colour inside the circle.
    Returns (coin_type_str, confidence_0_to_1, colour_tag).
    Thai coins:
      10 Baht — bi-metallic: gold/brass outer ring  → yellowish H 15-40
       5 Baht — nickel, occasionally copper edge     → silver/gray + maybe slight copper
       1 Baht — smaller nickel                       → silver/gray
    """
    h_img, w_img = img_bgr.shape[:2]
    # Sample ring region (outer 60–100 % of radius) to catch the outer ring of 10 Baht
    mask_outer = np.zeros((h_img, w_img), dtype=np.uint8)
    mask_inner = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.circle(mask_outer, (cx, cy), r,          255, -1)
    cv2.circle(mask_inner, (cx, cy), int(r*0.6), 255, -1)
    ring_mask = cv2.subtract(mask_outer, mask_inner)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ring_pixels = hsv[ring_mask > 0]

    if len(ring_pixels) == 0:
        return "5 Baht (24 mm)", 0.4, "silver"

    H, S, V = ring_pixels[:, 0], ring_pixels[:, 1], ring_pixels[:, 2]

    # Gold / brass: H 10-40, S > 60, V > 80
    gold_mask   = (H >= 10) & (H <= 40) & (S > 60) & (V > 80)
    # Copper/red: H 0-10 or 160-180, S > 60, V > 60
    copper_mask = ((H <= 10) | (H >= 160)) & (S > 60) & (V > 60)

    gold_ratio   = gold_mask.sum()   / len(ring_pixels)
    copper_ratio = copper_mask.sum() / len(ring_pixels)

    if gold_ratio > 0.12:
        conf = min(gold_ratio * 5, 1.0)
        return "10 Baht (26 mm)", round(conf, 2), "gold"
    elif copper_ratio > 0.08:
        conf = min(copper_ratio * 6, 0.85)
        return "5 Baht (24 mm)", round(conf, 2), "silver"
    else:
        # Distinguish 5 vs 1 by relative radius in the image
        rel = r / min(h_img, w_img)
        if rel > 0.085:
            return "5 Baht (24 mm)", 0.55, "silver"
        else:
            return "1 Baht (20 mm)", 0.55, "silver"


def detect_coin(img_bgr: np.ndarray):
    """
    Detect largest coin-like circle and classify it.
    Returns (circle_array, coin_type_str, confidence, colour_tag) or (None,…).
    """
    h, w = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    circles = _hough_circles_multi(gray, h, w)

    if not circles:
        return None, None, 0.0, "silver"

    # Pick the largest circle (most likely the coin, egg contours aren't circles)
    best = max(circles, key=lambda c: c[2])
    cx, cy, r = int(best[0]), int(best[1]), int(best[2])
    coin_type, conf, ctag = _classify_coin_by_color(img_bgr, cx, cy, r)
    return np.array([cx, cy, r], dtype=int), coin_type, conf, ctag


def detect_coin_manual(img_bgr: np.ndarray):
    """
    Detect largest circle only (no colour classification, type chosen by user).
    Returns circle array or None.
    """
    h, w = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    circles = _hough_circles_multi(gray, h, w)
    if not circles:
        return None
    return np.array(max(circles, key=lambda c: c[2]), dtype=int)


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

    st.markdown(f"""
    <div class="result-card">
      <div class="result-header">
        <h3>🏆 {label}</h3>
        <p>a = {a_input:.2f} mm &nbsp;|&nbsp; b_max = {b_max_input:.2f} mm</p>
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
            ellipse   = detect_egg_ellipse(img_bgr)

            if coin_mode == "🔍 Auto-detect coin type":
                coin_arr, detected_type, conf, ctag = detect_coin(img_bgr)
            else:
                coin_arr   = detect_coin_manual(img_bgr)
                detected_type = manual_coin_type
                conf, ctag = 1.0, "silver"

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
            ellipse2  = detect_egg_ellipse(img_bgr2)
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
        a_manual = st.number_input("a — half-length, end-to-end (mm)",
                                   min_value=1.0, max_value=100.0, value=30.0, step=0.1, key="a_man")
    with col_b:
        b_max_manual = st.number_input("b_max — maximum radius (mm)",
                                       min_value=1.0, max_value=80.0,  value=22.0, step=0.1, key="bmax_man")

    size_manual = st.selectbox("Egg grade (Thai standard)", list(EGG_PARAMS.keys()),
                               format_func=lambda x: EGG_PARAMS[x]["label"], key="size_man")
    st.caption("💡 *a* = half of the long axis. *b_max* = maximum equatorial radius.")

    if st.button("▶ Run Precision Analysis", key="run_manual"):
        try:
            results3 = compute_egg_metrics(float(a_manual), float(b_max_manual), size_manual)
            show_results(results3, size_manual, float(a_manual), float(b_max_manual))
        except Exception as e:
            st.error(f"Calculation error: {e}")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Mathematical model adapted from Narushin-type ovoid equations. "
    "Image processing: multi-approach edge fusion (CLAHE·Canny, bilateral·Canny, morphological gradient) "
    "with multi-parameter Hough circle detection and HSV colour-based coin classification. "
    "Thai egg grades 0–3 (TFDA standard). All volumes displayed in ml (1 ml = 1000 mm³)."
)
