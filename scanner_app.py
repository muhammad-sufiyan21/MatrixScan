import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="MatrixScan", page_icon="📄", layout="wide")

st.title("📄 MatrixScan")
st.caption("Document scanner powered by Linear Algebra")
st.write(
    "Upload a tilted document, set its four corners, correct the perspective, "
    "enhance the result, and download it as a PDF."
)

uploaded = st.file_uploader(
    "Upload a document photo",
    type=["jpg", "jpeg", "png"]
)

def order_points(points):
    """Return points in top-left, top-right, bottom-right, bottom-left order."""
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)],       # top-left
        pts[np.argmin(diff)],    # top-right
        pts[np.argmax(s)],      # bottom-right
        pts[np.argmax(diff)]    # bottom-left
    ], dtype=np.float32)

def make_pdf(image_rgb):
    """Convert an RGB image into downloadable one-page PDF bytes."""
    pil_image = Image.fromarray(image_rgb).convert("RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="PDF", resolution=150.0)
    return buffer.getvalue()

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if bgr is None:
        st.error("The image could not be read. Please upload another JPG or PNG.")
        st.stop()

    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    st.subheader("1. Original document")
    st.image(image, use_container_width=True)

    st.subheader("2. Set the four document corners")
    st.info(
        "Enter the pixel coordinates of the four corners. "
        "Use the original image above as a guide. Coordinates must be within the image."
    )

    default_points = {
        "Top-left X": int(width * 0.15),
        "Top-left Y": int(height * 0.15),
        "Top-right X": int(width * 0.85),
        "Top-right Y": int(height * 0.15),
        "Bottom-right X": int(width * 0.85),
        "Bottom-right Y": int(height * 0.85),
        "Bottom-left X": int(width * 0.15),
        "Bottom-left Y": int(height * 0.85),
    }

    cols = st.columns(4)
    labels = list(default_points.keys())
    values = {}
    for i, label in enumerate(labels):
        with cols[i % 4]:
            values[label] = st.number_input(
                label, min_value=0, max_value=max(width, height),
                value=min(default_points[label], max(width, height)),
                step=1
            )

    corners = [
        [values["Top-left X"], values["Top-left Y"]],
        [values["Top-right X"], values["Top-right Y"]],
        [values["Bottom-right X"], values["Bottom-right Y"]],
        [values["Bottom-left X"], values["Bottom-left Y"]],
    ]

    valid = all(0 <= x < width and 0 <= y < height for x, y in corners)
    if not valid:
        st.error(f"All points must be inside the image ({width} × {height}).")
        st.stop()

    if st.button("✨ Correct perspective", type="primary"):
        src = order_points(corners)

        tl, tr, br, bl = src
        out_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        out_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        out_width = max(out_width, 100)
        out_height = max(out_height, 100)

        destination = np.array([
            [0, 0],
            [out_width - 1, 0],
            [out_width - 1, out_height - 1],
            [0, out_height - 1]
        ], dtype=np.float32)

        # Linear Algebra: calculate a 3 x 3 homography matrix from four point pairs.
        H = cv2.getPerspectiveTransform(src, destination)

        # Apply the transformation matrix to remap the image into a rectangle.
        corrected = cv2.warpPerspective(image, H, (out_width, out_height))

        st.session_state["corrected"] = corrected
        st.session_state["matrix"] = H

    if "corrected" in st.session_state:
        corrected = st.session_state["corrected"]
        H = st.session_state["matrix"]

        st.subheader("3. Corrected document")
        left, right = st.columns(2)
        with left:
            st.image(corrected, caption="Perspective-corrected image", use_container_width=True)
        with right:
            st.markdown("**Calculated 3 × 3 homography matrix**")
            st.dataframe(np.round(H, 4), use_container_width=True)

        st.subheader("4. Enhance and export")
        mode = st.selectbox(
            "Choose an enhancement",
            ["Original colour", "Grayscale", "Black & white"]
        )

        if mode == "Grayscale":
            output = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        elif mode == "Black & white":
            gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            output = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        else:
            output = corrected

        st.image(output, caption=f"Final result: {mode}", use_container_width=True)

        pdf_bytes = make_pdf(output)
        st.download_button(
            "⬇️ Download scanned PDF",
            data=pdf_bytes,
            file_name="MatrixScan_document.pdf",
            mime="application/pdf"
        )
        st.download_button(
            "⬇️ Download corrected image",
            data=cv2.imencode(".png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))[1].tobytes(),
            file_name="MatrixScan_document.png",
            mime="image/png"
        )
else:
    st.info("Upload a JPG or PNG image to begin.")
    st.markdown(
        "**How it works:** Upload → enter four corner coordinates → "
        "calculate the homography matrix → correct → enhance → download PDF."
    )

st.divider()
st.caption("MatrixScan is an educational demonstration of perspective transformation. "
           "It does not automatically detect document corners.")
