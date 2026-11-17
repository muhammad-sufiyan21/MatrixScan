import io
import re

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="MatrixScan",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- STYLING --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --pink:#f36b91; --rose:#e94370; --plum:#8c3d82; --muted:#b9aebd; }
.stApp {
  background:
    radial-gradient(ellipse at 12% 0%, rgba(140,61,130,.25), transparent 38%),
    radial-gradient(ellipse at 90% 20%, rgba(233,67,112,.15), transparent 34%),
    linear-gradient(145deg,#100b17 0%,#09070e 65%,#130a16 100%);
  color:#f8f3f8; font-family:'DM Sans',sans-serif;
}
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1120px; padding-top:2rem; padding-bottom:4rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#fff !important; }
p, label, .stMarkdown { color:#eee6ef; }
.brand { font-family:'Space Grotesk'; font-weight:800; letter-spacing:-1.8px;
  font-size:clamp(2.5rem,6vw,4.8rem); line-height:1.02;
  background:linear-gradient(110deg,#fff 5%,#ff9db7 48%,#bd79c5 95%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.tagline { color:#c5b8c9; font-size:1.08rem; max-width:650px; line-height:1.7; }
.eyebrow { color:#ff9ab5; text-transform:uppercase; letter-spacing:2px; font-size:.75rem; font-weight:800; }
.panel { background:linear-gradient(145deg,rgba(255,255,255,.075),rgba(255,255,255,.025));
  border:1px solid rgba(255,255,255,.11); border-radius:22px; padding:22px;
  box-shadow:0 18px 55px rgba(0,0,0,.22); }
.feature { background:rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.09);
  border-radius:18px; padding:18px; min-height:125px; }
.mascot { display:inline-flex; align-items:center; justify-content:center; width:94px;height:94px;
  border-radius:28px; font-size:3.2rem; background:linear-gradient(145deg,#432044,#21152e);
  border:1px solid rgba(255,255,255,.16); box-shadow:0 15px 45px rgba(233,67,112,.2);
  animation:floaty 3s ease-in-out infinite; }
@keyframes floaty { 0%,100%{transform:translateY(0) rotate(-2deg)} 50%{transform:translateY(-10px) rotate(2deg)} }
.scan-stage { border:1px solid rgba(255,255,255,.13); background:#17101e; border-radius:20px;
  padding:18px; overflow:hidden; text-align:center; }
.scan-paper { width:min(250px,70%); height:310px; margin:14px auto; border-radius:9px;
  background:linear-gradient(180deg,#f8f5f6,#d8d0d8); position:relative; overflow:hidden;
  box-shadow:0 15px 40px rgba(0,0,0,.35); }
.scan-paper:before { content:''; position:absolute; left:0;right:0;height:5px;top:-8px;
  background:#ed5e91; box-shadow:0 0 18px 7px rgba(237,94,145,.55);
  animation:beam 2.1s ease-in-out infinite; }
@keyframes beam { 0%{top:0} 100%{top:100%} }
.scan-lines { position:absolute; inset:42px 25px; background:repeating-linear-gradient(
  to bottom,rgba(54,43,57,.28) 0 3px,transparent 3px 19px); }
.stButton>button, .stDownloadButton>button {
  border:0 !important; border-radius:13px !important; min-height:45px;
  background:linear-gradient(110deg,#a34891,#ec527b) !important; color:white !important;
  font-weight:700 !important; box-shadow:0 8px 24px rgba(233,67,112,.22);
  transition:transform .2s ease,filter .2s ease;
}
.stButton>button:hover,.stDownloadButton>button:hover { transform:translateY(-2px);filter:brightness(1.1); }
.stTextInput input,.stSelectbox div[data-baseweb="select"],.stTextArea textarea {
  background:#1a1320 !important; color:#fff !important; border-color:#49334d !important; border-radius:11px !important;
}
[data-testid="stFileUploader"] { background:rgba(255,255,255,.035); border:1px dashed #865276; border-radius:16px; padding:12px; }
hr { border-color:rgba(255,255,255,.12) !important; }
.small-muted { color:#b9aebd; font-size:.88rem; }
[data-testid="stSidebar"] { background:#100b17; }
</style>
""", unsafe_allow_html=True)

# -------------------- HELPERS --------------------
def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9 _-]", "", (name or "").strip())
    name = re.sub(r"\s+", "_", name).strip("_")
    return (name[:70] or "Scanned_Document")

def image_to_rgb(fileobj, max_side=2600):
    """Read an uploaded image, fix phone rotation (EXIF) and limit the size."""
    fileobj.seek(0)
    img = Image.open(fileobj)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return np.array(img)


def _order_points(pts):
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    rect[1] = pts[np.argmin(d)]   # top-right
    rect[3] = pts[np.argmax(d)]   # bottom-left
    return rect


def _quad_from_contour(cnt):
    """Turn any contour into the best 4-corner polygon."""
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in (0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
    # fallback: tilted bounding rectangle
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype("float32")


def _valid_quad(quad, w, h, min_frac=0.12):
    area = cv2.contourArea(quad.reshape(-1, 1, 2))
    if area < min_frac * w * h:
        return False
    return cv2.isContourConvex(quad.reshape(-1, 1, 2).astype("int32"))


def find_document_corners(rgb):
    """Return 4 corners (TL, TR, BR, BL) in original-image pixels, or None."""
    H, W = rgb.shape[:2]
    scale = 800.0 / max(H, W)
    small = cv2.resize(rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    candidates = []

    # Method 1: edges (works when the page has a clear border)
    med = float(np.median(blur))
    edges = cv2.Canny(blur, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    edges = cv2.dilate(edges, k, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    cs, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates += sorted(cs, key=cv2.contourArea, reverse=True)[:5]

    # Method 2: paper is brighter than the table (Otsu threshold)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for m in (th, 255 - th):
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates += sorted(cs, key=cv2.contourArea, reverse=True)[:3]

    # Method 3: paper is low-saturation (white/grey) while the table is coloured
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (7, 7), 0)
    _, th2 = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, k, iterations=3)
    cs, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates += sorted(cs, key=cv2.contourArea, reverse=True)[:3]

    # Method 4: GrabCut, seeded with "bright = paper, very dark = background"
    try:
        sm = cv2.GaussianBlur(gray, (0, 0), 3)
        gm = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        gm[sm > np.percentile(sm, 60)] = cv2.GC_PR_FGD
        gm[sm < np.percentile(sm, 20)] = cv2.GC_BGD
        cv2.grabCut(small, gm, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
        fgm = ((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)).astype(np.uint8) * 255
        fgm = cv2.morphologyEx(fgm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=2)
        cs, _ = cv2.findContours(fgm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates += sorted(cs, key=cv2.contourArea, reverse=True)[:2]
    except cv2.error:
        pass

    margin = 0.015 * max(w, h)

    def border_hits(q):
        return int(sum((x < margin or y < margin or x > w - margin or y > h - margin) for x, y in q))

    def edge_contrast(q):
        """Paper is clearly brighter inside than in a band just outside its edges."""
        poly = q.astype(np.int32).reshape(-1, 1, 2)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [poly], 255)
        r = max(5, int(0.025 * max(w, h)))
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        inner = cv2.erode(m, kk)
        inner_ring = cv2.subtract(m, inner)
        outer_ring = cv2.subtract(cv2.dilate(m, kk), m)
        if inner_ring.sum() == 0 or outer_ring.sum() == 0:
            return 0.0
        return float(gray[inner_ring > 0].mean() - gray[outer_ring > 0].mean())

    best, best_score, best_c = None, 0.0, None
    for c in candidates:
        if cv2.contourArea(c) < 0.12 * w * h:
            continue
        quad = _quad_from_contour(c)
        if not _valid_quad(quad, w, h):
            continue
        a = cv2.contourArea(quad.reshape(-1, 1, 2))
        hits = border_hits(quad)
        if hits >= 3 or a > 0.985 * w * h:   # just the photo frame, not a page
            continue
        contrast = edge_contrast(quad)
        if contrast < 12:                    # edge between marble and marble, not a page
            continue
        score = (a / (w * h)) * min(contrast, 80) * (1.0 - 0.2 * hits)
        if score > best_score:
            best, best_score, best_c = quad, score, c
    if best is None:
        return None
    cut_by_frame = False
    if _max_angle_error(best) > 20:           # page is cut by the photo edge -> keep it undistorted
        hull = cv2.convexHull(best_c)
        best = cv2.boxPoints(cv2.minAreaRect(hull)).astype("float32")
        cut_by_frame = True
    rect = _order_points(best / scale)
    if cut_by_frame:
        rect = _trim_to_right_page(rgb, rect)  # drop the cut-off side page, keep the main page
    c = rect.mean(axis=0)                      # trim ~1.5% inside so no table is left at the borders
    return (c + (rect - c) * 0.985).astype("float32")



def _find_spine_fraction(norm):
    """Notebook spread cut by the photo edge: find where the right page begins (0..1 of width), else None."""
    H, W = norm.shape
    small_w = 700
    sc = small_w / W
    n = cv2.resize(norm, (small_w, int(H * sc)), interpolation=cv2.INTER_AREA)
    dens = cv2.GaussianBlur((n < 175).astype(np.float32).mean(axis=0).reshape(1, -1), (0, 0), 2).ravel()
    win = int(0.12 * small_w)
    lo, hi = int(0.12 * small_w), int(0.70 * small_w)
    best_x, best_step = None, 0.0
    for x in range(lo, hi):
        left = dens[max(0, x - win):x].mean()
        right = dens[x:x + win].mean()
        step = right - left
        if step > best_step and right > 2.0 * max(left, 1e-4):
            best_x, best_step = x, step
    if best_x is None or best_step < 0.03:
        return None
    return max(0.0, (best_x - 0.012 * small_w) / small_w)


def _trim_to_right_page(rgb, rect):
    """If the left edge of the rectangle is only a cut-off page, move it to the spine."""
    try:
        w = _warp(rgb, rect)
        g = cv2.cvtColor(w, cv2.COLOR_RGB2GRAY)
        f = _find_spine_fraction(_flatten_light(g))
    except cv2.error:
        return rect
    if f is None or f < 0.12:
        return rect
    tl, tr, br, bl = rect
    tl, bl = tl + f * (tr - tl), bl + f * (br - bl)          # left edge -> spine
    tl, tr = tl + 0.045 * (bl - tl), tr + 0.045 * (br - tr)   # a little off the top (table/bookmark)
    tr, br = tr + 0.015 * (tl - tr), br + 0.015 * (bl - br)   # and the right edge
    return np.array([tl, tr, br, bl], dtype="float32")


def _max_angle_error(q):
    q = _order_points(q)
    worst = 0.0
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        worst = max(worst, abs(np.degrees(np.arccos(np.clip(cosang, -1, 1))) - 90))
    return worst


def _warp(rgb, rect):
    tl, tr, br, bl = rect
    wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
    out_w, out_h = int(max(wA, wB)), int(max(hA, hB))
    # keep output at a sensible size
    longest = max(out_w, out_h)
    if longest > 2200:
        s = 2200 / longest
        out_w, out_h = int(out_w * s), int(out_h * s)
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(rgb, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def _flatten_light(channel):
    """Remove shadows / uneven light by dividing by the page background."""
    h, w = channel.shape
    k = max(31, (min(h, w) // 12) | 1)
    bg = cv2.morphologyEx(channel, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, 21)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    norm = cv2.divide(channel, np.maximum(bg, 1), scale=255)
    return norm


def scan_image(rgb: np.ndarray, mode: str, corners=None) -> np.ndarray:
    """corners: optional 4 points (TL, TR, BR, BL) in pixels chosen by the user; otherwise auto-detect."""
    if corners is None:
        corners = find_document_corners(rgb)
    if corners is not None:
        warped = _warp(rgb, _order_points(np.asarray(corners, dtype="float32")))
    else:
        warped = rgb.copy()          # no document found: keep the full photo, no stretching

    if mode == "Color Original":
        ch = [_flatten_light(warped[:, :, i]) for i in range(3)]
        out = cv2.merge(ch)
        out = cv2.convertScaleAbs(out, alpha=1.08, beta=-8)   # little extra contrast
        return out

    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    norm = _flatten_light(gray)
    norm = cv2.fastNlMeansDenoising(norm, None, h=6, templateWindowSize=7, searchWindowSize=21)

    if mode == "Grayscale":
        # push the paper to pure white, keep the ink dark
        lo, hi = np.percentile(norm, 2), np.percentile(norm, 90)
        g = np.clip((norm.astype("float32") - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype("uint8")
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    # Black & White: CamScanner-style. Ink goes solid black, paper pure white, faint ruled lines stay light.
    f = norm.astype("float32")
    g = np.clip((f - 70.0) / (205.0 - 70.0), 0, 1)
    g = g ** 1.35                                   # slightly darker midtones -> sturdier handwriting
    bw = (g * 255).astype("uint8")
    blur = cv2.GaussianBlur(bw, (0, 0), 1.2)
    bw = cv2.addWeighted(bw, 1.6, blur, -0.6, 0)      # gentle sharpening
    m = max(6, min(bw.shape) // 120)                # clean white margin, hides leftover edge shadow
    bw[:m, :] = 255; bw[-m:, :] = 255; bw[:, :m] = 255; bw[:, -m:] = 255
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

def page_data(upload, key):
    """Read the photo once per upload and remember its auto-detected corners."""
    cache = st.session_state.setdefault("page_cache", {})
    if key not in cache:
        rgb = image_to_rgb(upload)
        cache[key] = (rgb, find_document_corners(rgb))
    return cache[key]

def draw_outline(rgb, corners, max_side=760):
    """Small preview of the photo with a pink outline around the part that will be scanned."""
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    prev = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if corners is not None:
        pts = (np.asarray(corners, dtype="float32") * s).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(prev, [pts], True, (236, 82, 123), 4)
        for p in pts.reshape(-1, 2):
            cv2.circle(prev, (int(p[0]), int(p[1])), 9, (255, 255, 255), -1)
            cv2.circle(prev, (int(p[0]), int(p[1])), 9, (236, 82, 123), 3)
    return prev

def crop_controls(rgb, auto, key):
    """Auto outline by default; optional sliders to move the four corners by hand."""
    h, w = rgb.shape[:2]
    manual = st.toggle("✂️ Adjust crop myself", key=f"manual_{key}")
    corners = auto
    if manual:
        if auto is not None:
            start = np.asarray(auto, dtype="float32")
        else:
            start = np.float32([[.05 * w, .05 * h], [.95 * w, .05 * h], [.95 * w, .95 * h], [.05 * w, .95 * h]])
        names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
        pts = []
        row1, row2 = st.columns(2), st.columns(2)
        for idx, col in enumerate(row1 + row2):
            order = [0, 1, 3, 2][idx]            # layout: TL TR / BL BR
            with col:
                st.markdown(f"**{names[order]}**")
                x0 = float(np.clip(start[order][0] / w * 100, 0, 100))
                y0 = float(np.clip(start[order][1] / h * 100, 0, 100))
                xp = st.slider("Left ↔ Right", 0.0, 100.0, x0, 0.5, key=f"{key}_x{order}")
                yp = st.slider("Up ↕ Down", 0.0, 100.0, y0, 0.5, key=f"{key}_y{order}")
            pts.append((order, [xp / 100 * w, yp / 100 * h]))
        pts.sort(key=lambda t: t[0])
        corners = np.float32([p for _, p in pts])
    st.image(draw_outline(rgb, corners), use_container_width=True)
    if corners is None:
        st.caption("No page edges found automatically. The whole photo will be used. Turn on “Adjust crop myself” to select the page.")
    elif not manual:
        st.caption("Pink outline = the part that will be scanned. Wrong? Turn on “Adjust crop myself”.")
    return corners if (manual or auto is not None) else None

def make_pdf(images: list[np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    page_w, page_h = letter
    for arr in images:
        pil = Image.fromarray(arr).convert("RGB")
        img_buffer = io.BytesIO()
        pil.save(img_buffer, format="JPEG", quality=92)
        img_buffer.seek(0)
        iw, ih = pil.size
        scale = min((page_w - 48) / iw, (page_h - 48) / ih)
        draw_w, draw_h = iw * scale, ih * scale
        pdf.drawImage(ImageReader(img_buffer), (page_w - draw_w) / 2, (page_h - draw_h) / 2,
                      width=draw_w, height=draw_h, preserveAspectRatio=True)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()

def set_page(page):
    st.session_state.page = page

def init_state():
    defaults = {
        "page": "landing", "processed_pages": [], "pdf_bytes": None,
        "pdf_name": "My_Scan", "last_uploads": None, "notice": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# -------------------- TOP BAR --------------------
def topbar():
    left, right = st.columns([4, 1])
    with left:
        st.markdown("### ✦ MatrixScan")
    with right:
        if st.session_state.page != "landing":
            if st.button("Home", use_container_width=True):
                set_page("landing")
                st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)

# -------------------- LANDING --------------------
if st.session_state.page == "landing":
    st.markdown('<div class="eyebrow">THE DOCUMENT SCANNER, REIMAGINE
