import io
import os
import re
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from supabase import create_client, Client

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="MatrixScan",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "")
BUCKET = "matrixscan-pdfs"

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
def get_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("Supabase settings are missing. Add SUPABASE_URL and SUPABASE_ANON_KEY to Streamlit secrets.")
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

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
    if _max_angle_error(best) > 20:           # page is cut by the photo edge -> keep it undistorted
        hull = cv2.convexHull(best_c)
        best = cv2.boxPoints(cv2.minAreaRect(hull)).astype("float32")
    return _order_points(best / scale)


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


def scan_image(rgb: np.ndarray, mode: str) -> np.ndarray:
    corners = find_document_corners(rgb)
    if corners is not None:
        # trim ~1.5% inside the edge so no table/background is left at the borders
        c = corners.mean(axis=0)
        corners = (c + (corners - c) * 0.985).astype("float32")
        warped = _warp(rgb, corners)
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
        "page": "landing", "user": None, "access_token": None,
        "refresh_token": None, "processed_pages": [], "pdf_bytes": None,
        "pdf_name": "My_Scan", "last_uploads": None, "notice": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

def user_id():
    return st.session_state.user["id"]

def save_pdf_to_cloud(pdf_bytes: bytes, name: str):
    sb = get_supabase()
    uid = user_id()
    file_id = f"{uuid.uuid4().hex}.pdf"
    path = f"{uid}/{file_id}"
    sb.storage.from_(BUCKET).upload(
        path, pdf_bytes,
        file_options={"content-type": "application/pdf", "upsert": "false"}
    )
    row = {
        "user_id": uid, "name": name, "storage_path": path,
        "page_count": len(st.session_state.processed_pages),
    }
    result = sb.table("scans").insert(row).execute()
    return result.data[0]

def list_scans():
    sb = get_supabase()
    return sb.table("scans").select("*").eq("user_id", user_id()).order("created_at", desc=True).execute().data

def signed_pdf_url(path: str):
    sb = get_supabase()
    result = sb.storage.from_(BUCKET).create_signed_url(path, 60 * 60)
    return result.get("signedURL") or result.get("signedUrl")

def logout():
    try:
        if st.session_state.access_token:
            get_supabase().auth.sign_out()
    except Exception:
        pass
    for key in ["user", "access_token", "refresh_token"]:
        st.session_state[key] = None
    st.session_state.page = "landing"
    st.rerun()

# -------------------- TOP BAR --------------------
def topbar():
    left, right = st.columns([4, 1])
    with left:
        st.markdown("### ✦ MatrixScan")
    with right:
        if st.session_state.user:
            if st.button("Log out", use_container_width=True):
                logout()
    st.markdown("<hr>", unsafe_allow_html=True)

# -------------------- LANDING --------------------
if st.session_state.page == "landing" and not st.session_state.user:
    st.markdown('<div class="eyebrow">THE DOCUMENT SCANNER, REIMAGINED</div>', unsafe_allow_html=True)
    a, b = st.columns([1.55, 1], vertical_alignment="center")
    with a:
        st.markdown('<div class="brand">MatrixScan.</div>', unsafe_allow_html=True)
        st.markdown('<p class="tagline">Turn everyday photos into clean, polished documents. Scan in color, grayscale or crisp black & white — then keep every PDF in your own library.</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Get started  →", use_container_width=True):
                set_page("auth")
                st.rerun()
        with c2:
            if st.button("Explore features", use_container_width=True):
                st.session_state.show_features = True
        st.markdown('<p class="small-muted">Simple document scanning. Beautifully organized.</p>', unsafe_allow_html=True)
    with b:
        st.markdown("""
        <div class="panel" style="text-align:center;padding:32px 18px;">
          <div class="mascot">🤖</div>
          <h2 style="margin-top:18px;">Ready when you are.</h2>
          <p class="small-muted">Your little scanning assistant is waiting to help.</p>
          <div style="display:flex;gap:8px;justify-content:center;margin-top:18px;">
            <span style="background:#34203a;padding:8px 12px;border-radius:30px;">✦ Smart scan</span>
            <span style="background:#34203a;padding:8px 12px;border-radius:30px;">✦ PDF export</span>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    for col, emoji, title, desc in [
        (f1, "📷", "Scan your way", "Choose color, grayscale or black & white."),
        (f2, "📚", "Multi-page PDFs", "Combine several images into one document."),
        (f3, "☁️", "Your own library", "Save scans privately to your account.")
    ]:
        with col:
            st.markdown(f'<div class="feature"><div style="font-size:1.6rem">{emoji}</div><b>{title}</b><p class="small-muted">{desc}</p></div>', unsafe_allow_html=True)

# -------------------- AUTH --------------------
elif st.session_state.page == "auth" and not st.session_state.user:
    topbar()
    st.markdown('<div class="eyebrow">YOUR PRIVATE DOCUMENT SPACE</div>', unsafe_allow_html=True)
    st.title("Welcome to MatrixScan")
    st.markdown('<p class="tagline">Sign in or create an account to keep your PDFs safe and accessible.</p>', unsafe_allow_html=True)
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error("Authentication is not configured yet. Follow SETUP.md to connect Supabase before using login.")
    mode = st.segmented_control("Account", ["Log in", "Create account"], default="Log in")
    email = st.text_input("Email address", placeholder="you@example.com")
    password = st.text_input("Password", type="password", placeholder="At least 8 characters")
    if mode == "Create account":
        if st.button("Create my account", use_container_width=True):
            if not email or "@" not in email or len(password) < 8:
                st.error("Enter a valid email and a password with at least 8 characters.")
            else:
                try:
                    response = get_supabase().auth.sign_up({"email": email, "password": password})
                    if response.user and response.session:
                        st.session_state.user = {"id": response.user.id, "email": response.user.email}
                        st.session_state.access_token = response.session.access_token
                        st.session_state.refresh_token = response.session.refresh_token
                        set_page("dashboard")
                        st.rerun()
                    else:
                        st.success("Account created. Check your email to confirm it, then log in.")
                except Exception as e:
                    st.error(f"Could not create account: {e}")
    else:
        if st.button("Log in securely", use_container_width=True):
            try:
                response = get_supabase().auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.user = {"id": response.user.id, "email": response.user.email}
                st.session_state.access_token = response.session.access_token
                st.session_state.refresh_token = response.session.refresh_token
                set_page("dashboard")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed. Check your details and try again. ({e})")
    if st.button("← Back to welcome"):
        set_page("landing")
        st.rerun()

# -------------------- DASHBOARD --------------------
elif st.session_state.user and st.session_state.page == "dashboard":
    topbar()
    email = st.session_state.user["email"]
    display = email.split("@")[0].replace(".", " ").replace("_", " ").title()
    st.markdown(f'<div class="eyebrow">YOUR WORKSPACE</div><h1>Welcome back, {display} ✦</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Everything you scan, organized in one place.</p>', unsafe_allow_html=True)
    if st.button("＋  Create a new scan", use_container_width=True):
        st.session_state.processed_pages = []
        st.session_state.pdf_bytes = None
        set_page("scanner")
        st.rerun()
    st.markdown("## My Scans")
    try:
        scans = list_scans()
        if not scans:
            st.markdown('<div class="panel" style="text-align:center"><div style="font-size:2.3rem">📄</div><h3>Your library is ready</h3><p class="small-muted">Your saved PDFs will appear here. Create your first scan to get started.</p></div>', unsafe_allow_html=True)
        else:
            for scan in scans:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="center")
                    with c1:
                        st.markdown(f"**📄 {scan['name']}.pdf**")
                        st.caption(f"{scan.get('page_count', 1)} page(s) · {scan.get('created_at', '')[:10]}")
                    with c2:
                        if st.button("Share", key=f"share_{scan['id']}"):
                            try:
                                url = signed_pdf_url(scan["storage_path"])
                                st.session_state[f"url_{scan['id']}"] = url
                            except Exception as e:
                                st.error(f"Could not create share link: {e}")
                    with c3:
                        if st.button("Delete", key=f"delete_{scan['id']}"):
                            try:
                                sb = get_supabase()
                                sb.storage.from_(BUCKET).remove([scan["storage_path"]])
                                sb.table("scans").delete().eq("id", scan["id"]).eq("user_id", user_id()).execute()
                                st.success("Deleted.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Delete failed: {e}")
                    url = st.session_state.get(f"url_{scan['id']}")
                    if url:
                        st.code(url, language=None)
                        wa = "https://wa.me/?text=" + __import__("urllib.parse").parse.quote(f"My MatrixScan PDF: {url}")
                        st.link_button("Share on WhatsApp", wa)
                        st.link_button("Open / download PDF", url)
    except Exception as e:
        st.error(f"Could not load your library: {e}")
        st.info("Check that the Supabase SQL setup and private storage bucket have been created.")

# -------------------- SCANNER --------------------
elif st.session_state.user and st.session_state.page == "scanner":
    topbar()
    if st.button("← Dashboard"):
        set_page("dashboard")
        st.rerun()
    st.markdown('<div class="eyebrow">STEP 01 · ADD YOUR PAGES</div>', unsafe_allow_html=True)
    st.title("Let's scan something.")
    uploads = st.file_uploader("Choose one or more document photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    mode = st.radio("Choose your scan style", ["Black & White", "Grayscale", "Color Original"], horizontal=True)
    if uploads:
        st.markdown(f"**{len(uploads)} page(s) selected**")
        preview_cols = st.columns(min(3, len(uploads)))
        for i, up in enumerate(uploads):
            with preview_cols[i % len(preview_cols)]:
                st.image(up, caption=f"Page {i+1}", use_container_width=True)
        if st.button("Start scanning  →", use_container_width=True):
            st.markdown('<div class="scan-stage"><div class="eyebrow">MATRIXSCAN ENGINE</div><h3>Scanning your pages…</h3><div class="scan-paper"><div class="scan-lines"></div></div><p class="small-muted">Aligning · Enhancing · Preparing your PDF</p></div>', unsafe_allow_html=True)
            progress = st.progress(0)
            processed = []
            try:
                for i, up in enumerate(uploads):
                    up.seek(0)
                    rgb = image_to_rgb(up)
                    processed.append(scan_image(rgb, mode))
                    progress.progress(int((i + 1) / len(uploads) * 100))
                st.session_state.processed_pages = processed
                st.session_state.pdf_bytes = make_pdf(processed)
                st.session_state.pdf_name = "My_Scan"
                set_page("results")
                st.rerun()
            except Exception as e:
                st.error(f"Scanning failed: {e}")
    else:
        st.markdown('<div class="panel" style="text-align:center"><div style="font-size:2rem">＋</div><b>Add your document photos</b><p class="small-muted">You can select multiple images to make a multi-page PDF.</p></div>', unsafe_allow_html=True)

# -------------------- RESULTS --------------------
elif st.session_state.user and st.session_state.page == "results":
    topbar()
    if st.button("← Back to scanner"):
        set_page("scanner")
        st.rerun()
    st.markdown('<div class="eyebrow">STEP 02 · REVIEW & EXPORT</div>', unsafe_allow_html=True)
    st.title("Your scan is ready. ✨")
    pages = st.session_state.processed_pages
    if pages:
        st.success(f"Successfully scanned {len(pages)} page(s).")
        for i, page in enumerate(pages):
            with st.expander(f"Preview page {i+1}", expanded=(i == 0)):
                st.image(page, use_container_width=True)
        st.markdown("### Name your PDF")
        name = st.text_input("PDF filename (without .pdf)", value=st.session_state.pdf_name)
        st.session_state.pdf_name = safe_filename(name)
        pdf_bytes = st.session_state.pdf_bytes or make_pdf(pages)
        st.download_button("↓  Download PDF", data=pdf_bytes,
                           file_name=f"{st.session_state.pdf_name}.pdf",
                           mime="application/pdf", use_container_width=True)
        if st.button("☁️  Save to My Scans", use_container_width=True):
            try:
                with st.spinner("Saving securely to your account…"):
                    save_pdf_to_cloud(pdf_bytes, st.session_state.pdf_name)
                st.success("Saved to your private MatrixScan library.")
                set_page("dashboard")
                st.rerun()
            except Exception as e:
                st.error(f"Could not save PDF: {e}")
                st.info("Check Supabase setup and storage policies in SETUP.md.")
        st.caption("Tip: use your browser's share menu after downloading, or save the PDF and attach it in WhatsApp/email.")
    else:
        st.warning("No scanned pages found. Go back and upload images.")

# -------------------- FALLBACK --------------------
else:
    st.session_state.page = "landing"
    st.rerun()
