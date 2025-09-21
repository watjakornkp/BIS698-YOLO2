import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch

st.set_page_config(page_title="YOLO Image Detection", layout="wide")
st.title("YOLO Image Detection App")
st.markdown("---")

# Load YOLO model
@st.cache_resource
def load_model():
    """Load the YOLO model and cache it to prevent re-loading on every interaction."""
    return YOLO("yolo11n.pt") # Ensure this model path is correct
    # If you have a custom model, use the correct path, e.g., YOLO("runs/detect/train73/weights/best.pt")

model = load_model()

# Upload image
st.subheader("1. Upload Your Image")
uploaded_image = st.file_uploader("Choose an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Use st.columns to display images side by side for a better UI
    col1, col2 = st.columns(2)

    with col1:
        st.info("Uploaded Image")
        st.image(uploaded_image, use_container_width=True)

    # Convert uploaded image to a format YOLO can use
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)

    # Prepare the image for the model
    # Convert from NumPy array to PyTorch tensor
    # Change dimensions from (H, W, C) to (C, H, W)
    # Add a batch dimension (1, C, H, W)
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).unsqueeze(0)

    # Check the data type and convert to float if necessary (YOLO expects float)
    if image_tensor.dtype != torch.float32:
        image_tensor = image_tensor.float()

    st.subheader("2. Run Object Detection")
    run_detection = st.button("Run YOLO Detection")
    st.markdown("---")
    
    if run_detection:
        with st.spinner("Processing... This may take a moment."):
            try:
                # Run YOLO inference on the correctly formatted tensor
                results = model.predict(image_tensor, conf=0.4)
                
                with col2:
                    st.success("Detection Completed!")
                    result_image = results[0].plot()
                    st.image(result_image, caption="YOLO Detection Result", use_container_width=True)

                # Count the number of detected people
                boxes = results[0].boxes
                class_ids = boxes.cls.cpu().numpy().astype(int)
                class_names = [model.names[i] for i in class_ids]
                person_count = class_names.count("person")
                
                st.subheader("3. Detection Summary")
                st.write(f"Number of objects detected: **{len(class_names)}**")
                st.write(f"Number of people detected: **{person_count}**")

            except Exception as e:
                st.error(f"An error occurred during detection: {e}")
                st.warning("Please try uploading another image or check the model file path.")
