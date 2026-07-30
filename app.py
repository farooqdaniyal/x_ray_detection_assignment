# ============================================
# app.py — X-Ray Fracture Detection (Streamlit)
# ============================================
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="X-Ray Fracture Detection",
    page_icon="🦴",
    layout="wide"
)

# ============================================
# CUSTOM CSS — Radiology Viewer Theme
# (same design language as the Gradio version)
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.stApp {
    background: radial-gradient(ellipse 900px 500px at 50% -10%, rgba(63,221,224,0.06), transparent 60%), #0A0D10;
}

h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #E7ECEF !important;
}

.stMarkdown, p, label {
    font-family: 'IBM Plex Mono', monospace !important;
}

div[data-testid="stMetric"] {
    background: #12161B;
    border: 1px solid #232B33;
    border-radius: 6px;
    padding: 12px;
}

.stButton > button {
    background: #3FDDE0 !important;
    color: #04191A !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border: none !important;
    border-radius: 4px !important;
}
.stButton > button:hover {
    background: #5FEAEC !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL (cached so it only loads once)
# ============================================
@st.cache_resource
def load_model():
    return YOLO('xray_model.pt')

xray_model = load_model()

# ============================================
# HEADER
# ============================================
st.title("🦴 Fracture Detection")
st.caption("Radiographic bone-fracture screening, powered by YOLOv11")
st.divider()

# ============================================
# LAYOUT
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Input Scan**")
    uploaded_file = st.file_uploader("Upload Radiograph", type=['jpg', 'jpeg', 'png'])
    confidence = st.slider("Detection Sensitivity", 0.1, 0.9, 0.25, 0.05)
    run_btn = st.button("Run Detection", use_container_width=True)

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)

with col2:
    st.markdown("**Detection Readout**")

    if run_btn:
        if uploaded_file is None:
            st.warning("⚠️ Please upload an image first!")
        else:
            with st.spinner("Analyzing..."):
                image_np = np.array(image)
                results = xray_model.predict(source=image_np, conf=confidence, verbose=False)
                output = results[0].plot()
                count = len(results[0].boxes)
                confs = [round(float(c), 2) for c in results[0].boxes.conf]

            st.image(output, caption="Annotated Result", use_container_width=True)

            if count > 0:
                st.error(f"STATUS: FINDINGS PRESENT\n\n{count} fracture site(s) detected\n\nConfidence: {confs}")
            else:
                st.success("STATUS: CLEAR\n\nNo fracture patterns detected in this study.")
    else:
        st.info("Awaiting input — upload a radiograph and click **Run Detection**.")

st.divider()
st.caption("⚠️ **Disclaimer:** This system is for educational purposes only and is not a diagnostic device. Always consult a qualified medical professional.")