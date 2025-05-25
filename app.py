import streamlit as st
from PIL import Image
import numpy as np
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from ultralytics import YOLO
import pathlib

# === Load Models ===
clot_model = load_model("blood_group/clot_classifier_model.h5")
sickle_model = load_model("sickle_cell_model_mobnetv2.h5")
thalassemia_model = YOLO("Thalassemia.pt")

# Handle PosixPath issue on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
leukemia_model = torch.hub.load("ultralytics/yolov5", "custom", path="bestluk.pt", force_reload=True)
pathlib.PosixPath = temp

# === Clot Detection ===
def is_clotted(img):
    img = img.convert("RGB").resize((128, 128))
    arr = keras_image.img_to_array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = clot_model.predict(arr)[0][0]
    return pred < 0.5

def predict_clot(image):
    if image is None:
        return "❌ No image uploaded."
    label = "Clotted" if is_clotted(image) else "Non-Clotted"
    return f"🩸 Clot Detection: *{label}*"

# === Sickle Cell Detection ===
def predict_sickle(image):
    if image is None:
        return "❌ No image uploaded."
    img = image.convert("RGB").resize((128, 128))
    arr = keras_image.img_to_array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = sickle_model.predict(arr)[0][0]
    label = "Normal" if pred >= 0.5 else "Sickle Cell"
    confidence = pred if pred >= 0.5 else 1 - pred
    return f"🧬 Sickle Cell Detection: *{label}*\nConfidence: {confidence:.2f}"

# === Blood Group Identification ===
def split_into_three_drops(image):
    width, height = image.size
    third = width // 3
    return (
        image.crop((0, 0, third, height)),
        image.crop((third, 0, 2 * third, height)),
        image.crop((2 * third, 0, width, height))
    )

def determine_blood_group(a_clot, b_clot, d_clot):
    if a_clot and b_clot:
        base = "AB"
    elif a_clot:
        base = "A"
    elif b_clot:
        base = "B"
    else:
        base = "O"
    rh = "+" if d_clot else "-"
    return base + rh

def detect_blood_group(image):
    drop_a, drop_b, drop_d = split_into_three_drops(image)
    a_clot = is_clotted(drop_a)
    b_clot = is_clotted(drop_b)
    d_clot = is_clotted(drop_d)
    group = determine_blood_group(a_clot, b_clot, d_clot)
    return drop_a, drop_b, drop_d, f"🧪 Blood Group: *{group}*\n\nClotting:\n- Anti-A: {'Yes' if a_clot else 'No'}\n- Anti-B: {'Yes' if b_clot else 'No'}\n- Anti-D: {'Yes' if d_clot else 'No'}"

# === Thalassemia Classification ===
def predict_thalassemia(image):
    results = thalassemia_model(image, save=False)[0]
    top_class_idx = int(results.probs.top1)
    top_class_label = thalassemia_model.names[top_class_idx]
    confidence = results.probs.data[top_class_idx]
    return f"🧬 Thalassemia Classification: *{top_class_label}*\nConfidence: {confidence:.2f}"

# === Leukemia Detection ===
class_info = {
    'Benign': "Stage: Benign cells. No immediate concern, monitor routinely.",
    'Early': "Stage: Early leukemia detected. Please consult a hematologist.",
    'Pre': "Stage: Pre-leukemic condition. Immediate medical consultation recommended.",
    'Pro': "Stage: Pro-leukemic (progressed). Urgent specialist intervention required."
}

def predict_leukemia(image):
    image_path = "temp_leukemia.jpg"
    image.save(image_path)
    results = leukemia_model(image_path)
    predictions = results.pandas().xyxy[0]
    if predictions.empty:
        return "🧬 No abnormal cells detected. Please check image quality."
    labels = predictions['name'].value_counts().to_dict()
    report = "🧬 *Leukemia Detection Summary*:\n"
    for cls, count in labels.items():
        report += f"\n- *{cls}*: {count} cell(s)\n  → {class_info.get(cls, 'No info available.')}"
    return report

# === Streamlit App ===
st.title("🩸 Blood Group & Disease Detection System")
task = st.selectbox("Select Task", [
    "Check Clotness",
    "Blood Group Identification",
    "Sickle Cell Detection",
    "Thalassemia Detection",
    "Leukemia Detection"
])

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)


    if task == "Check Clotness":
        if st.button("🔍 Predict Clot"):
            st.markdown(predict_clot(image))

    elif task == "Blood Group Identification":
        if st.button("🔍 Predict Blood Group"):
            a, b, d, result = detect_blood_group(image)
            cols = st.columns(3)
            for col, img, cap in zip(cols, [a, b, d], ["Anti-A", "Anti-B", "Anti-D"]):
                with col:
                    st.image(img, caption=cap, use_container_width=True)

            st.markdown(result)

    elif task == "Sickle Cell Detection":
        if st.button("🔍 Predict Sickle Cell"):
            st.markdown(predict_sickle(image))

    elif task == "Thalassemia Detection":
        if st.button("🔍 Predict Thalassemia"):
            st.markdown(predict_thalassemia(image))

    elif task == "Leukemia Detection":
        if st.button("🔍 Predict Leukemia"):
            st.markdown(predict_leukemia(image))
