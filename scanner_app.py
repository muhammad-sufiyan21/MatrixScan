import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time

# Page configuration
st.set_page_config(
    page_title="MatrixScan - AI Document Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Burgundy & Pink Glassmorphism Theme (SaaS Style)
st.markdown("""
<style>
    /* Global Dark Burgundy Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #1a0b1c 0%, #0d040f 50%, #1f0a1a 100%);
        color: #fce7f3;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Glassmorphism Cards */
    .saas-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(233, 64, 87, 0.2);
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }

    /* Glowing Headings */
    .main-title {
        font-size: 3.2rem !important;
        font-weight: 800;
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 50%, #e94057 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 35px rgba(233, 64, 87, 0.4);
        margin-bottom: 0px;
    }

    .sub-title {
        color: #f472b6;
        font-size: 1.15rem;
        font-weight: 500;
        margin-bottom: 25px;
    }

    /* Gradient Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #e94057 0%, #8a2387 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 20px rgba(233, 64, 87, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(233, 64, 87, 0.7) !important;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'my_scans' not in st.session_state:
    st.session_state.my_scans = []
if 'current_scan' not in st.session_state:
    st.session_state.current_scan = None

# ================= PAGE 1: LOGIN / WELCOME =================
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;"><h1 class="main-title">MatrixScan</h1><p class="sub-title">Your Documents, Perfectly Scanned & Enhanced 🌸</p></div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.subheader("🔐 Welcome Back")
            email = st.text_input("Email Address", placeholder="sufyan@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            if st.button("Get Started / Login", use_container_width=True):
                if email:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.session_state.page = 'dashboard'
                    st.rerun()
                else:
                    st.warning("Please enter your email to continue.")
            st.markdown('</div>', unsafe_allow_html=True)

# ================= PAGE 2: DASHBOARD & APP FLOW =================
else:
    # Top Navigation Bar
    nav_col1, nav_col2, nav_col3 = st.columns([2, 2, 1])
    with nav_col1:
        st.markdown(f"### 👋 Welcome, {st.session_state.get('user_email', 'User')}")
    with nav_col3:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.page = 'login'
            st.rerun()

    st.markdown("---")

    # Sidebar Navigation / Quick Actions
    menu = st.radio("Navigation", ["📂 My Dashboard", "✨ New Scan"], horizontal=True)

    if menu == "📂 My Dashboard":
        st.markdown("### 🗂️ Your Saved Scans")
        
        if len(st.session_state.my_scans) == 0:
            st.info("No scans found yet! Click on **'New Scan'** above to create your first professional PDF document.")
        else:
            for idx, scan in enumerate(st.session_state.my_scans):
                st.markdown(f"""
                <div class="saas-card">
                    <h4>📄 {scan['name']}</h4>
                    <p style="color: #f472b6; margin: 0;">Created on: {scan['date']}</p>
                </div>
                """, unsafe_allow_html=True)
                col_d1, col_d2 = st.columns([1, 4])
                with col_d1:
                    st.download_button(
                        label="📥 Download",
                        data=scan['data'],
                        file_name=f"{scan['name']}.pdf",
                        mime="application/pdf",
                        key=f"dl_{idx}"
                    )

    elif menu == "✨ New Scan":
        st.markdown("### 📤 Upload Document for Smart Scanning")
        
        uploaded_file = st.file_uploader("Upload document image (JPG, PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_np = np.array(image)
            h, w = img_np.shape[:2] if len(img_np.shape) == 3 else img_np.shape

            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.image(image, caption="Original Document", use_container_width=True)
            
            with col2:
                st.markdown("### 🎨 Select Scan Style")
                scan_style = st.selectbox(
                    "Choose enhancement mode:",
                    ["Original Color", "Black & White (High Contrast)", "Clean Grayscale"]
                )
                
                st.markdown("### 📐 Corner Adjustment (Homography)")
                c1, c2 = st.columns(2)
                tl_x = c1.number_input("Top Left X", value=int(w * 0.1))
                tl_y = c2.number_input("Top Left Y", value=int(h * 0.1))
                
                c3, c4 = st.columns(2)
                tr_x = c3.number_input("Top Right X", value=int(w * 0.9))
                tr_y = c4.number_input("Top Right Y", value=int(h * 0.1))

                c5, c6 = st.columns(2)
                br_x = c5.number_input("Bottom Right X", value=int(w * 0.9))
                br_y = c6.number_input("Bottom Right Y", value=int(h * 0.9))

                c7, c8 = st.columns(2)
                bl_x = c7.number_input("Bottom Left X", value=int(w * 0.1))
                bl_y = c8.number_input("Bottom Left Y", value=int(h * 0.9))

                if st.button("🚀 Start Animated Scanning"):
                    # Animated Loading sequence matching user requirement
                    with st.status("✨ Processing Document...", expanded=True) as status:
                        st.write("Preparing your document...")
                        time.sleep(0.8)
                        st.write("Applying Homography Matrix & Perspective Correction...")
                        time.sleep(0.8)
                        st.write("Enhancing document quality...")
                        time.sleep(0.8)
                        status.update(label="Scanning Complete Successfully!", state="complete", expanded=False)

                    # Processing Image
                    pts1 = np.float32([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]])
                    width, height = 800, 1050
                    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
                    
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    scanned = cv2.warpPerspective(img_np, matrix, (width, height))

                    if "Black & White" in scan_style:
                        gray = cv2.cvtColor(scanned, cv2.COLOR_RGB2GRAY)
                        scanned = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        scanned = cv2.cvtColor(scanned, cv2.COLOR_GRAY2RGB)
                    elif "Grayscale" in scan_style:
                        gray = cv2.cvtColor(scanned, cv2.COLOR_RGB2GRAY)
                        scanned = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

                    st.session_state.processed_image = scanned
                    st.success("🎉 Document Scanned Successfully!")

        # Show result preview & PDF saving if processed
        if 'processed_image' in st.session_state:
            st.markdown("---")
            st.markdown("### 🎯 Scanned Result & Export")
            st.image(st.session_state.processed_image, caption="Final Scanned Document", use_container_width=True)

            pdf_name = st.text_input("Enter Custom PDF Filename", value="University Assignment")

            scanned_pil = Image.fromarray(st.session_state.processed_image)
            img_byte_arr = io.BytesIO()
            scanned_pil.save(img_byte_arr, format='PDF')
            pdf_data = img_byte_arr.getvalue()

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    label="📥 Download PDF Now",
                    data=pdf_data,
                    file_name=f"{pdf_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            with col_b:
                if st.button("💾 Save to My Dashboard", use_container_width=True):
                    import datetime
                    new_scan_entry = {
                        "name": pdf_name,
                        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "data": pdf_data
                    }
                    st.session_state.my_scans.append(new_scan_entry)
                    st.success(f"'{pdf_name}.pdf' saved securely to your dashboard!")
