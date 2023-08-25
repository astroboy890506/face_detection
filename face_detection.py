import streamlit as st
import cv2
from matplotlib import pyplot as plt

def detect_and_mark_objects(image_path, cascade_xml_path, min_size=(20, 20)):
    # Open the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load the cascade classifier
    cascade_classifier = cv2.CascadeClassifier(cascade_xml_path)

    # Detect objects using the cascade classifier
    found = cascade_classifier.detectMultiScale(img_gray, minSize=min_size)

    # Draw rectangles around the detected objects
    for (x, y, width, height) in found:
        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (0, 255, 0), 5)

    return img_rgb

def main():
    st.set_page_config(
        page_title="Object Detection App",
        page_icon=":camera:",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("Object Detection using Cascade Classifier")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        cascade_xml_path = st.text_input("Enter the path to the cascade XML file")

        if cascade_xml_path and st.button("Detect Objects"):
            detected_image = detect_and_mark_objects(image_path, cascade_xml_path)
            image_width = st.slider("Adjust Image Width", min_value=100, max_value=800, value=500)
            st.subheader("Image with Detected Objects")
            st.image(detected_image, width=image_width)

if __name__ == "__main__":
    main()
