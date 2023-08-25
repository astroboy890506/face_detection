import streamlit as st
import cv2
from matplotlib import pyplot as plt

def detect_faces(image_path):
    # Opening image
    img = cv2.imread(image_path)

    # OpenCV opens images as BGR but we want it as RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use minSize to ignore small detections
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    found = face_cascade.detectMultiScale(img_gray, minSize=(20, 20))

    # Draw rectangles around detected faces
    for (x, y, w, h) in found:
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 5)

    return img_rgb

def main():
    st.set_page_config(
        page_title="Face Detection App",
        page_icon=":camera:",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("Face Detection")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        detected_image = detect_faces(image_path)
        
        image_width = st.slider("Adjust Image Width", min_value=100, max_value=800, value=500)

        st.subheader("Original Image")
        st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), width=image_width)

if __name__ == "__main__":
    main()
