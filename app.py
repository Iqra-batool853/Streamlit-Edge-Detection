import cv2
import numpy as np
import streamlit as st
from PIL import Image

# --- Custom CSS Styling ---
st.markdown("""
<style>
/* Page background */
main {
    background-color: #F5F7FA;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #E3EAF2;
    color: #001F3F;
}

/* Titles */
h1, h2, h3 {
    color: #003366;
    font-family: 'Poppins', sans-serif;
}

/* Buttons */
.stButton > button {
    background-color: #4A90E2;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5em 1em;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #003366;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border: 2px dashed #4A90E2;
    border-radius: 10px;
    padding: 15px;
}
main {
    background: linear-gradient(135deg, #E3F2FD, #F5F7FA);
}

/* Headers & captions */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)
# --- App Title and Description ---
st.markdown("<h1 style='text-align:center; color:#003366;'> Interactive Edge Detection App Using Streamlit and OpenCV</h1>", unsafe_allow_html=True)

st.markdown("""
<div style="
    background-color: #FFFFFF;
    border-left: 5px solid #4A90E2;
    padding: 15px 20px;
    margin-top: 10px;
    border-radius: 5px;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
">
    <h4 style="color:#003366;">Description:</h4>
    <p style="font-size:16px; color:#333333;">
        This application provides an easy-to-use interface for understanding the effects of 
        various <b>edge detection techniques</b> such as <b>Sobel</b>, <b>Laplacian</b>, and <b>Canny</b>.<br>
    </p>
</div>
""", unsafe_allow_html=True)
 
# --- Image Upload ---
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Read image
    image = np.array(Image.open(uploaded_file))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" Original Image")
        st.image(image, width='stretch')

    # --- Algorithm Selection ---
    st.sidebar.header("⚙️ Choose Algorithm & Parameters")
    algorithm = st.sidebar.selectbox("Select Edge Detection Algorithm", ("Sobel", "Laplacian", "Canny"))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Algorithm Parameters ---
    if algorithm == "Sobel":
        st.sidebar.subheader("Sobel Parameters")
        ksize = st.sidebar.slider("Kernel Size (odd number)", 1, 9, 3, step=2)
        direction = st.sidebar.radio("Gradient Direction", ("X", "Y", "Both"))
        if st.sidebar.button("Apply"):
            if direction == "X":
                edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            elif direction == "Y":
                edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            else:
                gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
                gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
                edges = cv2.magnitude(gx, gy)
            edges = cv2.convertScaleAbs(edges)

    elif algorithm == "Laplacian":
        st.sidebar.subheader("Laplacian Parameters")
        ksize = st.sidebar.slider("Kernel Size (odd number)", 1, 9, 3, step=2)
        if st.sidebar.button("Apply"):
            edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
            edges = cv2.convertScaleAbs(edges)

    elif algorithm == "Canny":
        st.sidebar.subheader("Canny Parameters")
        lower = st.sidebar.slider("Lower Threshold", 0, 255, 50)
        upper = st.sidebar.slider("Upper Threshold", 0, 255, 150)
        blur = st.sidebar.slider("Gaussian Blur (odd number)", 1, 9, 3, step=2)
        if st.sidebar.button("Apply"):
            blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
            edges = cv2.Canny(blurred, lower, upper)

    # --- Display Output ---
    if "edges" in locals():
        with col2:
            st.subheader(" Edge Detection Output")
            st.image(edges, width='stretch', clamp=True)
else:
    st.info("Please upload an image to start.")
