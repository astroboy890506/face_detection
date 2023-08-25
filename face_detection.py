import streamlit as st
import cv2

def detect_faces(image_path, cascade_model):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found = cascade_model.detectMultiScale(img_gray, minSize=(20, 20))

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

    cascade_model = None

    cascade_file = st.file_uploader("Upload a Cascade Model (XML)", type=["xml"])

    if cascade_file is not None:
        cascade_model = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_file.name)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        if cascade_model:
            detected_image = detect_faces(image_path, cascade_model)
            image_width = st.slider("Adjust Image Width", min_value=100, max_value=800, value=500)
            st.subheader("Original Image with Detected Faces")
            st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), width=image_width)

if __name__ == "__main__":
    main()
