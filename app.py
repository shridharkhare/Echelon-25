import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Streamlit App Config
st.set_page_config(page_title="P&ID Analyzer", layout="wide")
st.title("Automated Pipeline & I&C Detection")

# Upload Image
uploaded_file = st.file_uploader("Upload a P&ID Image", type=["png", "jpg"])

if uploaded_file is not None:
    # Read and Preprocess Image
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Edge Detection (Canny) + Pipeline Detection (Hough Lines)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10
    )

    # Detect Circles (Valves/Instruments)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=30
    )

    # OCR for Labels
    custom_config = r"--oem 3 --psm 6"
    text_data = pytesseract.image_to_string(gray, config=custom_config)

    # Draw Results
    result_img = img.copy()

    # Draw Detected Lines (Pipelines)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw Detected Circles (Instruments)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv2.circle(result_img, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)

    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            img, caption="Original Image", use_container_width=True
        )  # Updated parameter
    with col2:
        st.image(
            result_img, caption="Detected Elements", use_container_width=True
        )  # Updated parameter

    # Show Extracted Text
    st.subheader("Detected Labels")
    st.code(text_data)
