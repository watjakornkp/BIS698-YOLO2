import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.title("YOLO Image Detection App")

# โหลดโมเดล
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")
model = load_model()

# อัปโหลดรูปภาพ
uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # แสดงรูปภาพต้นฉบับ
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # อ่านรูปภาพและแปลงเป็น NumPy array
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)

    # รันการทำนายผล
    st.info("Running YOLO object detection...")
    
    # ใช้ predict() โดยตรง ซึ่งจะจัดการการปรับขนาดให้อัตโนมัติ
    results = model.predict(image_np, conf=0.4, imgsz=640)

    # วาดผลลัพธ์ลงบนรูปภาพ
    result_image = results[0].plot()
    st.image(result_image, caption="YOLO Detection Result", use_container_width=True)
    st.success("Detection completed!")
    
    # นับจำนวนคน
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    class_names = [model.names[i] for i in class_ids]
    person_count = class_names.count("person")
    st.write(f"Number of people detected: **{person_count}**")
