import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import io

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thai Egg Sorter",
    page_icon="🥚",
    layout="centered",
)

# ─── CSS Styling ──────────────────────────────────────────────────────────────
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
    display: flex;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid #eee;
    font-size: 14px;
  }
  .metric-label { color: #555; font-weight: 500; }
  .metric-value { color: #4A90E2; font-weight: 600; }

  .section-title {
    font-size: 13px;
    font-weight: 700;
    color: #333;
    margin: 16px 0 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .badge {
    display: inline-block;
    background: #e8f4fd;
    color: #4A90E2;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 4px;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 16px;
  }
  .detect-box {
    background: #f8fafc;
    border: 1px dashed #c0cfe0;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
    font-size: 13px;
    color: #555;
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

# ─── Thai Egg Size Constants ──────────────────────────────────────────────────
EGG_PARAMS = {
    0: dict(g=0.21, t=27.55, shell_thick=0.053, white_pct=0.712, yolk_pct=0.288, w=69.43, p=1.0647, label="Grade 0 (Jumbo ≥65g)"),
    1: dict(g=0.21, t=26.90, shell_thick=0.050, white_pct=0.698, yolk_pct=0.302, w=67.07, p=1.0645, label="Grade 1 (Large 60–65g)"),
    2: dict(g=0.23, t=26.59, shell_thick=0.050, white_pct=0.682, yolk_pct=0.318, w=61.02, p=1.0639, label="Grade 2 (Medium 55–60g)"),
    3: dict(g=0.22, t=25.14, shell_thick=0.060, white_pct=0.684, yolk_pct=0.307, w=58.21, p=1.0636, label="Grade 3 (Small <55g)"),
}

COIN_DIAMETERS_MM = {"10 Baht (26 mm)": 26.0, "5 Baht (24 mm)": 24.0, "1 Baht (20 mm)": 20.0}

EGG_SHELL_VOLUME_RATIO = 0.065
VOLUME_OL_AIR = 0.05


# ─── Core Math (all equations from original file preserved) ──────────────────
def compute_egg_metrics(a: float, b_max: float, size: int) -> dict:
    ep = EGG_PARAMS[size]
    g   = ep["g"]
    t   = ep["t"]
    p   = ep["p"]
    white_pct = ep["white_pct"]
    yolk_pct  = ep["yolk_pct"]
    shell_thick = ep["shell_thick"]

    # Adjusted b
    b = (b_max * a) / (2 * a - t)

    # k parameter
    k = -2 * (g + ((g ** 3) / 2)) / b

    # ── Volume ────────────────────────────────────────────────────────────────
    V = (
        ((2 / 9) * math.pi * a * (b ** 2))
        * (
            (10 * ((1 / (9 - (6 * k * a))) + (1 / (9 - (6 * k * a)))))
            + (8 * ((1 / (9 - (3 * k * a))) + (1 / (9 + (3 * k * a)))))
            + 1
        )
    )

    # ── Arc Length (Circumference) ─────────────────────────────────────────────
    def safe_sqrt(x):
        return math.sqrt(x) if x >= 0 else 0

    ArcLength = (math.pi / 18) * (
        (b * ((1 / math.sqrt(1 + k * a)) - (1 / math.sqrt(1 + k * a))))
        + 2 * (
            (safe_sqrt(a ** 2 + ((b ** 2) * (7 * k * a - 4 * math.sqrt(3)) ** 2) / (2 * (2 - math.sqrt(3) * k * a)) ** 3))
            + (safe_sqrt(a ** 2 + ((b ** 2) * (7 * k * a + 4 * math.sqrt(3)) ** 2) / (2 * (2 + math.sqrt(3) * k * a)) ** 3))
            + (a * math.sqrt(1 + ((b ** 2) * (k ** 2)) / 4))
        )
        + (safe_sqrt(3 * a ** 2 + ((b ** 2) * (5 * k * a - 4) ** 2) / (2 * (2 - k * a)) ** 3))
        + (safe_sqrt(3 * a ** 2 + ((b ** 2) * (5 * k * a + 4) ** 2) / (2 * (2 + k * a)) ** 3))
    )

    # ── Surface Area ───────────────────────────────────────────────────────────
    surface_area = S = (
        ((2 * math.pi * a * b) / 9)
        * (
            4 * (math.sqrt((3 / (3 - 2 * k * a)) * (5 / 9 + (b ** 2 * ((13 * k * a - 12) ** 2)) / (12 * (3 - 2 * k * a) ** 3))))
            + (math.sqrt((3 / (3 + 2 * k * a)) * (5 / 9 + (b ** 2 * ((13 * k * a + 12) ** 2)) / (12 * (3 + 2 * k * a) ** 3))))
            + 2 * (math.sqrt((3 / (3 - k * a)) * (8 / 9 + (b ** 2 * ((10 * k * a - b) ** 2)) / (12 * (3 - k * a) ** 3))))
            + (math.sqrt((3 / (3 + k * a)) * (8 / 9 + (b ** 2 * ((10 * k * a + b) ** 2)) / (12 * (3 + k * a) ** 3))))
            + (
                math.sqrt(1 + (a ** 2 * b ** 2 * k ** 2) / 4)
                - ((2 * a * b * k) / (1 - a ** 2 * k ** 2))
            )
        )
    )

    # Shorter surface area approximation
    surface_area_approx = (a / 2) * (
        (b / a) * (1 / (1 + k * a) - 1 / (1 - k * a))
        + 2 * safe_sqrt(1 + ((b ** 2) * (k ** 2)) / 4)
    )

    # ── Derived volumes ───────────────────────────────────────────────────────
    shell_volume   = surface_area * EGG_SHELL_VOLUME_RATIO
    usable_volume  = V * (1 - VOLUME_OL_AIR)
    edible_volume  = usable_volume - shell_volume

    amount_of_egg_white   = white_pct * edible_volume
    amount_of_egg_yolk    = yolk_pct  * edible_volume
    amount_of_albumin     = amount_of_egg_white * ((0.59 * p) / 1.03)
    amount_of_folic_acid  = edible_volume * (((0.47 * 1e-6) / 1.6) * p)
    amount_of_iron        = edible_volume * (((0.0172 * 1e-3) / 7.8) * p)
    amount_of_choline     = edible_volume * (((2.85 * 1e-3) / 1.09) * p)
    amount_of_zinc        = edible_volume * (((0.011 * 1e-3) / 7.14) * p)
    amount_of_vit_b12     = edible_volume * (((0.027 * 1e-6) / 0.52) * p)

    return dict(
        b=b, k=k, g=g, t=t, shell_thick=shell_thick, p=p,
        V=V, surface_area=surface_area, surface_area_approx=surface_area_approx,
        ArcLength=ArcLength, shell_volume=shell_volume,
        usable_volume=usable_volume, edible_volume=edible_volume,
        white_pct=white_pct, yolk_pct=yolk_pct,
        egg_white=amount_of_egg_white, egg_yolk=amount_of_egg_yolk,
        albumin=amount_of_albumin, folic_acid=amount_of_folic_acid,
        iron=amount_of_iron, choline=amount_of_choline,
        zinc=amount_of_zinc, vit_b12=amount_of_vit_b12,
    )


# ─── Image Processing ─────────────────────────────────────────────────────────
def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def detect_egg_ellipse(img_bgr: np.ndarray):
    """Returns (cx, cy, major_ax, minor_ax, angle) or None."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges  = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0
    h, w = img_bgr.shape[:2]

    for cnt in contours:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if area < (h * w * 0.01) or area > (h * w * 0.85):
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (ma, mi), angle = ellipse
        if mi < 1:
            continue
        ratio = ma / mi
        if ratio < 1.1 or ratio > 2.5:        # egg-like aspect ratio
            continue
        score = area
        if score > best_score:
            best_score = score
            best = ellipse

    return best  # ((cx,cy),(MA,mi),angle)  or None


def detect_coin_circle(img_bgr: np.ndarray):
    """Returns (cx, cy, radius_px) of largest coin-like circle, or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = img_bgr.shape[:2]

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min(h, w) * 0.1,
        param1=60, param2=35,
        minRadius=int(min(h, w) * 0.04),
        maxRadius=int(min(h, w) * 0.30),
    )
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype(int)
    # Return largest circle (by radius)
    return max(circles, key=lambda c: c[2])   # (cx, cy, r)


def annotate_image(img_bgr: np.ndarray, ellipse, coin) -> np.ndarray:
    out = img_bgr.copy()
    if ellipse is not None:
        cv2.ellipse(out, ellipse, (0, 200, 60), 3)
        cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
        cv2.circle(out, (cx, cy), 5, (0, 200, 60), -1)
        cv2.putText(out, "EGG", (cx - 20, cy - int(ellipse[1][0] / 2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 60), 2)
    if coin is not None:
        cx, cy, r = coin
        cv2.circle(out, (cx, cy), r, (255, 140, 0), 3)
        cv2.putText(out, "COIN", (cx - 25, cy - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 0), 2)
    return out


def cv_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


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
        st.metric("Volume (mm³)", f"{r['V']:.2f}")
        st.metric("Surface Area (mm²)", f"{r['surface_area']:.2f}")
        st.metric("Arc Length (mm)", f"{r['ArcLength']:.2f}")
        st.metric("Shell Volume (mm³)", f"{r['shell_volume']:.2f}")
    with col2:
        st.metric("Usable Volume (mm³)", f"{r['usable_volume']:.2f}")
        st.metric("Edible Volume (mm³)", f"{r['edible_volume']:.2f}")
        st.metric("b (adjusted, mm)", f"{r['b']:.4f}")
        st.metric("k", f"{r['k']:.6f}")

    with st.expander("🥚 Egg White & Yolk Breakdown"):
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Egg White (mm³)", f"{r['egg_white']:.2f}")
            st.metric("White Fraction", f"{r['white_pct']*100:.1f}%")
        with c2:
            st.metric("Egg Yolk (mm³)", f"{r['egg_yolk']:.2f}")
            st.metric("Yolk Fraction", f"{r['yolk_pct']*100:.1f}%")

    with st.expander("🧬 Nutritional Composition (volume-based)"):
        cols = st.columns(2)
        nutrients = [
            ("Albumin (mm³)",       r["albumin"]),
            ("Folic Acid (mm³)",    r["folic_acid"]),
            ("Iron (mm³)",          r["iron"]),
            ("Choline (mm³)",       r["choline"]),
            ("Zinc (mm³)",          r["zinc"]),
            ("Vitamin B12 (mm³)",   r["vit_b12"]),
        ]
        for i, (label_n, val) in enumerate(nutrients):
            with cols[i % 2]:
                st.metric(label_n, f"{val:.8f}")

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
            language="text"
        )


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📸 Photo + Coin", "📸 Photo + Grade", "⌨️ Manual Input"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PHOTO + COIN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("**Upload a photo of the egg with a Thai coin beside it for scale.**")
    coin_type = st.selectbox("Coin type", list(COIN_DIAMETERS_MM.keys()), key="coin_type")
    size_coin = st.selectbox("Egg grade (for nutrition lookup)", list(EGG_PARAMS.keys()),
                             format_func=lambda x: EGG_PARAMS[x]["label"], key="size_coin")
    uploaded_coin = st.file_uploader("Upload egg + coin image", type=["jpg", "jpeg", "png"], key="up_coin")

    if uploaded_coin:
        pil_img = Image.open(uploaded_coin)
        img_bgr = pil_to_cv(pil_img)

        with st.spinner("🔍 Detecting egg and coin..."):
            ellipse = detect_egg_ellipse(img_bgr)
            coin_c  = detect_coin_circle(img_bgr)
            annotated = annotate_image(img_bgr, ellipse, coin_c)

        st.image(cv_to_pil(annotated), caption="Detection result", use_container_width=True)

        if ellipse is None:
            st.warning("⚠️ Could not detect egg contour. Try a cleaner background or better lighting.")
        elif coin_c is None:
            st.warning("⚠️ Coin not detected. Place the coin clearly beside the egg.")
        else:
            coin_real_mm = COIN_DIAMETERS_MM[coin_type]
            coin_px      = coin_c[2] * 2          # diameter in pixels
            mm_per_px    = coin_real_mm / coin_px

            major_ax_px  = max(ellipse[1])        # long axis (pixels)
            minor_ax_px  = min(ellipse[1])        # short axis (pixels)

            a_val    = (major_ax_px * mm_per_px) / 2   # half-length = a
            b_max_val = (minor_ax_px * mm_per_px) / 2  # max radius = b_max

            st.markdown(f"""
            <div class="detect-box">
              ✅ <b>Coin diameter:</b> {coin_px:.1f} px → {coin_real_mm} mm
              &nbsp;|&nbsp; <b>Scale:</b> {mm_per_px:.4f} mm/px<br>
              🥚 <b>Egg:</b> long axis {major_ax_px:.1f} px ({major_ax_px*mm_per_px:.2f} mm),
              short axis {minor_ax_px:.1f} px ({minor_ax_px*mm_per_px:.2f} mm)<br>
              📐 <b>a</b> = {a_val:.2f} mm &nbsp;|&nbsp; <b>b_max</b> = {b_max_val:.2f} mm
            </div>
            """, unsafe_allow_html=True)

            if st.button("▶ Run Precision Analysis", key="run_coin"):
                try:
                    results = compute_egg_metrics(a_val, b_max_val, size_coin)
                    show_results(results, size_coin, a_val, b_max_val)
                except Exception as e:
                    st.error(f"Calculation error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PHOTO + GRADE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("**Upload a photo of the egg. Select the known Thai grade to scale dimensions.**")

    # Baseline long-axis lengths per grade (mm) used for proportional scaling
    GRADE_BASELINE_LENGTH_MM = {0: 65.0, 1: 61.0, 2: 57.5, 3: 54.0}

    known_grade = st.selectbox("Known egg grade", list(EGG_PARAMS.keys()),
                               format_func=lambda x: EGG_PARAMS[x]["label"], key="known_grade")
    uploaded_grade = st.file_uploader("Upload egg image (no coin needed)", type=["jpg", "jpeg", "png"], key="up_grade")

    if uploaded_grade:
        pil_img2 = Image.open(uploaded_grade)
        img_bgr2 = pil_to_cv(pil_img2)

        with st.spinner("🔍 Detecting egg shape..."):
            ellipse2 = detect_egg_ellipse(img_bgr2)
            annotated2 = annotate_image(img_bgr2, ellipse2, None)

        st.image(cv_to_pil(annotated2), caption="Detection result", use_container_width=True)

        if ellipse2 is None:
            st.warning("⚠️ Could not detect egg. Try better lighting or a plain background.")
        else:
            major_px = max(ellipse2[1])
            minor_px = min(ellipse2[1])
            ratio    = minor_px / major_px  # width-to-length ratio

            baseline_len_mm = GRADE_BASELINE_LENGTH_MM[known_grade]
            length_mm = baseline_len_mm
            width_mm  = baseline_len_mm * ratio

            a_val2    = length_mm / 2
            b_max_val2 = width_mm / 2

            st.markdown(f"""
            <div class="detect-box">
              🥚 <b>Detected pixel ratio</b> (w/l): {ratio:.4f}<br>
              📐 Scaled to Grade {known_grade} baseline: length = {length_mm:.2f} mm,
              width = {width_mm:.2f} mm<br>
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
        a_manual = st.number_input("a — half-length, end-to-end (mm)", min_value=1.0,
                                   max_value=100.0, value=30.0, step=0.1, key="a_man")
    with col_b:
        b_max_manual = st.number_input("b_max — maximum radius (mm)", min_value=1.0,
                                       max_value=80.0, value=22.0, step=0.1, key="bmax_man")

    size_manual = st.selectbox("Egg grade (Thai standard)", list(EGG_PARAMS.keys()),
                               format_func=lambda x: EGG_PARAMS[x]["label"], key="size_man")

    st.caption("💡 *a* = half of the long axis (top-to-bottom centre). *b_max* = maximum equatorial radius.")

    if st.button("▶ Run Precision Analysis", key="run_manual"):
        try:
            results3 = compute_egg_metrics(float(a_manual), float(b_max_manual), size_manual)
            show_results(results3, size_manual, float(a_manual), float(b_max_manual))
        except Exception as e:
            st.error(f"Calculation error: {e}")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Mathematical model adapted from original research equations (Narushin-type ovoid). "
    "Image processing via OpenCV contour & Hough circle detection. "
    "For Thai standard egg grades 0–3."
)
