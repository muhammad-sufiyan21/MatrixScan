import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Page config
st.set_page_config(
    page_title="MatrixScan - Modern Document Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism & Modern Aesthetic CSS
st.markdown("""
<style>
    /* Dark Aesthetic Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c20 0%, #150a19 50%, #05020a 100%);
        color: #f0f2f5;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Glassmorphism Cards */
    div[data-testid="stExpander"], div.stMarkdown > div {
        border-radius: 16px;
    }
    
    .css-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Glowing Gradient Title */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(135deg, #ff416c 0%, #8a2387 50%, #e94057 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(233, 64, 87, 0.3);
        margin-bottom: 5px;
    }

    .sub-title {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Styled Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #8a2387 0%, #e94057 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(233, 64, 87, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 64, 87, 0.6) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 12, 32, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">MatrixScan</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Linear Algebra Powered Document Scanner & PDF Generator</div>', unsafe_allow_html=True)

# Main UI layout using columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a crooked or angled photo", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Input Image", use_container_width=True)

with col2:
    st.markdown("### ⚙️ Transformation Settings")
    if uploaded_file is not None:
        st.info("Coordinates represent the 4 corners of your document in the photo.")
        
        # Default coordinates or user inputs
        img_np = np.array(image)
        h, w = img_np.shape[:0] if len(img_np.shape) == 3 else img_np.shape
        
        # Coordinates input UI
        st.write("**Select Corner Points (x, y):**")
        c1 = st.columns(2)
        top_left_x = c1[0].number_input("Top Left X", value=int(w * 0.1))
        top_left_y = c1[1].number_input("Top Left Y", value=int(h * 0.1))
        
        c2 = st.columns(2)
        top_right_x = c2[0].number_input("Top Right X", value=int(w * 0.9))
        top_right_y = c2[1].number_input("Top Right Y", value=int(h * 0.1))
        
        c3 = st.columns(2)
        bottom_right_x = c3[0].number_input("Bottom Right X", value=int(w * 0.9))
        bottom_right_y = c3[1].number_input("Bottom Right Y", value=int(h * 0.9))
        
        c4 = st.columns(2)
        bottom_left_x = c4[0].number_input("Bottom Left X", value=int(w * 0.1))
        bottom_left_y = c4[1].number_input("Bottom Left Y", value=int(h * 0.9))
        
        if st.button("✨ Apply Homography Matrix & Scan"):
            # Source points from user input
            pts1 = np.float32([
                [top_left_x, top_left_y],
                [top_right_x, top_right_y],
                [bottom_right_x, bottom_right_y],
                [bottom_left_x, bottom_left_y]
            ])
            
            # Target dimensions
            width, height = 800, 1050
            pts2 = np.float32([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height]
            ])
            
            # 3x3 Homography Matrix calculation
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            scanned = cv2.warpPerspective(img_np, matrix, (width, height))
            
            # Display Result
            st.markdown("---")
            st.markdown("### 🎯 Scanned Result")
            st.image(scanned, caption="Perspective Corrected Document", use_container_width=True)
            
            # Convert to PDF
            scanned_pil = Image.fromarray(scanned)
            img_byte_arr = io.BytesIO()
            scanned_pil.save(img_byte_arr, format='PDF')
            
            st.download_button(
                label="📥 Download Scanned PDF",
                data=img_byte_arr.getvalue(),
                file_name="MatrixScan_Document.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("Please upload a document photo on the left to activate settings.")
