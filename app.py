import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import Counter # เพิ่มการนำเข้า Counter

st.set_page_config(page_title="YOLO Image Detection", layout="wide")
st.title("YOLO Image Detection App")
st.markdown("---")

# Load YOLO model
@st.cache_resource
def load_model():
    """Load the YOLO model and cache it to prevent re-loading on every interaction."""
    return YOLO("yolo11n.pt")
    # If you have a custom model, use the correct path, e.g., YOLO("runs/detect/train73/weights/best.pt")

model = load_model()

# Upload image
st.subheader("1. Upload Your Image")
uploaded_image = st.file_uploader("Choose an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.info("Uploaded Image")
        st.image(uploaded_image, use_container_width=True)

    # Convert uploaded image to a format YOLO can use
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)

    st.subheader("2. Run Object Detection")
    run_detection = st.button("Run YOLO Detection")
    st.markdown("---")
    
    if run_detection:
        with st.spinner("Processing... This may take a moment."):
            try:
                # Run YOLO inference
                # ใช้ predict() โดยตรง ซึ่งจะจัดการการปรับขนาดให้อัตโนมัติ
                results = model.predict(image_np, conf=0.4, imgsz=640)
                
                with col2:
                    st.success("Detection Completed!")
                    result_image = results[0].plot()
                    st.image(result_image, caption="YOLO Detection Result", use_container_width=True)

                # Extract detection results and count all object types
                boxes = results[0].boxes
                class_ids = boxes.cls.cpu().numpy().astype(int)
                class_names = [model.names[i] for i in class_ids]
                
                # Use Counter to count all unique objects
                detection_counts = Counter(class_names)
                
                st.subheader("3. Detection Summary")
                
                # Display total number of objects
                st.write(f"Number of total objects detected: **{len(class_names)}**")
                
                # Display count for each object type
                if detection_counts:
                    st.write("---")
                    st.write("**Detected Objects:**")
                    for name, count in detection_counts.items():
                        st.write(f"- {name}: **{count}**")
                else:
                    st.warning("No objects were detected in the image.")

            except Exception as e:
                st.error(f"An error occurred during detection: {e}")
                st.warning("Please try uploading another image or check the model file path.")
