from PIL import Image
import numpy as np
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from ultralytics import YOLO
import gradio as gr

# === Load Models ===
clot_model = load_model("blood_group/clot_classifier_model.h5")
sickle_model = load_model("sickle_cell_model_mobnetv2.h5")
thalassemia_model = YOLO("Thalassemia.pt")  # YOLOv8 classifier
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

leukemia_model = torch.hub.load("ultralytics/yolov5", "custom", path="bestluk.pt", force_reload=True)

# Restore PosixPath
pathlib.PosixPath = temp

# === Clot Detection ===
def is_clotted(img):
    img = img.convert("RGB").resize((128, 128))
    arr = keras_image.img_to_array(img)
    arr = arr.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = clot_model.predict(arr)[0][0]
    return pred < 0.5  # True = clotted

def predict_clot(image):
    try:
        if image is None:
            return "❌ No image uploaded."
        label = "Clotted" if is_clotted(image) else "Non-Clotted"
        return f"🩸 Clot Detection: *{label}*"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# === Sickle Cell Detection ===
def predict_sickle(image):
    try:
        if image is None:
            return "❌ No image uploaded."
        img = image.convert("RGB").resize((128, 128))
        arr = keras_image.img_to_array(img)
        arr = arr.astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = sickle_model.predict(arr)[0][0]
        label = "Normal" if pred >= 0.5 else "Sickle Cell"
        confidence = pred if pred >= 0.5 else 1 - pred
        return f"🧬 Sickle Cell Detection: *{label}*\nConfidence: {confidence:.2f}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

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
    try:
        drop_a, drop_b, drop_d = split_into_three_drops(image)
        a_clot = is_clotted(drop_a)
        b_clot = is_clotted(drop_b)
        d_clot = is_clotted(drop_d)
        group = determine_blood_group(a_clot, b_clot, d_clot)
        return (
            drop_a, drop_b, drop_d,
            f"🧪 Blood Group: *{group}*\n\nClotting:\n- Anti-A: {'Yes' if a_clot else 'No'}\n- Anti-B: {'Yes' if b_clot else 'No'}\n- Anti-D: {'Yes' if d_clot else 'No'}"
        )
    except Exception as e:
        return None, None, None, f"❌ Error: {str(e)}"

# === Thalassemia Classification (YOLOv8) ===
def predict_thalassemia(image):
    try:
        results = thalassemia_model(image, save=False)[0]
        top_class_idx = int(results.probs.top1)
        top_class_label = thalassemia_model.names[top_class_idx]
        confidence = results.probs.data[top_class_idx]
        return f"🧬 Thalassemia Classification: *{top_class_label}*\nConfidence: {confidence:.2f}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# === Leukemia Detection (YOLOv5) ===
class_info = {
    'Benign': "Stage: Benign cells. No immediate concern, monitor routinely.",
    'Early': "Stage: Early leukemia detected. Please consult a hematologist.",
    'Pre': "Stage: Pre-leukemic condition. Immediate medical consultation recommended.",
    'Pro': "Stage: Pro-leukemic (progressed). Urgent specialist intervention required."
}

def predict_leukemia(image):
    try:
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

    except Exception as e:
        return f"❌ Error: {str(e)}"

# === Gradio GUI ===
with gr.Blocks(css="body { background-color: white; color: #111; }") as demo:
    gr.Markdown("## 🩸 Blood Group & Disease Detection System")
    gr.Markdown("Select a task, upload an image, and click Predict.")

    task_selector = gr.Dropdown(
        choices=[
            "Check Clotness",
            "Blood Group Identification",
            "Sickle Cell Detection",
            "Thalassemia Detection",
            "Leukemia Detection"
        ],
        label="Select Task",
        value=None,
        interactive=True
    )

    # Clotness Section
    with gr.Column(visible=False) as clot_section:
        clot_image = gr.Image(type="pil", label="Upload Blood Drop Image")
        clot_result = gr.Textbox(label="Prediction Result", lines=4)
        with gr.Row():
            clot_predict = gr.Button("🔍 Predict")
            clot_clear = gr.Button("🔄 Clear")
        clot_predict.click(fn=predict_clot, inputs=clot_image, outputs=clot_result)
        clot_clear.click(fn=lambda: (None, ""), outputs=[clot_image, clot_result])

    # Blood Group Section
    with gr.Column(visible=False) as group_section:
        group_image = gr.Image(type="pil", label="Upload 3-Drop Blood Image")
        with gr.Row():
            drop_a_img = gr.Image(label="Anti-A Drop", interactive=False)
            drop_b_img = gr.Image(label="Anti-B Drop", interactive=False)
            drop_d_img = gr.Image(label="Anti-D Drop", interactive=False)
        group_result = gr.Textbox(label="Prediction Result", lines=6)
        with gr.Row():
            group_predict = gr.Button("🔍 Predict Blood Group")
            group_clear = gr.Button("🔄 Clear")
        group_predict.click(fn=detect_blood_group, inputs=group_image, outputs=[drop_a_img, drop_b_img, drop_d_img, group_result])
        group_clear.click(fn=lambda: (None, None, None, None, ""), outputs=[group_image, drop_a_img, drop_b_img, drop_d_img, group_result])

    # Sickle Cell Section
    with gr.Column(visible=False) as sickle_section:
        sickle_image = gr.Image(type="pil", label="Upload Blood Cell Image")
        sickle_result = gr.Textbox(label="Prediction Result", lines=4)
        with gr.Row():
            sickle_predict = gr.Button("🔍 Predict")
            sickle_clear = gr.Button("🔄 Clear")
        sickle_predict.click(fn=predict_sickle, inputs=sickle_image, outputs=sickle_result)
        sickle_clear.click(fn=lambda: (None, ""), outputs=[sickle_image, sickle_result])

    # Thalassemia Section
    with gr.Column(visible=False) as thal_section:
        thal_image = gr.Image(type="pil", label="Upload Blood Image")
        thal_result = gr.Textbox(label="Prediction Result", lines=4)
        with gr.Row():
            thal_predict = gr.Button("🔍 Predict")
            thal_clear = gr.Button("🔄 Clear")
        thal_predict.click(fn=predict_thalassemia, inputs=thal_image, outputs=thal_result)
        thal_clear.click(fn=lambda: (None, ""), outputs=[thal_image, thal_result])

    # Leukemia Section
    with gr.Column(visible=False) as leukemia_section:
        leukemia_image = gr.Image(type="pil", label="Upload Microscopic Blood Image")
        leukemia_result = gr.Textbox(label="Prediction Result", lines=8)
        with gr.Row():
            leukemia_predict = gr.Button("🔍 Predict")
            leukemia_clear = gr.Button("🔄 Clear")
        leukemia_predict.click(fn=predict_leukemia, inputs=leukemia_image, outputs=leukemia_result)
        leukemia_clear.click(fn=lambda: (None, ""), outputs=[leukemia_image, leukemia_result])

    # Toggle UI Sections
    def show_section(task):
        return (
            gr.update(visible=task == "Check Clotness"),
            gr.update(visible=task == "Blood Group Identification"),
            gr.update(visible=task == "Sickle Cell Detection"),
            gr.update(visible=task == "Thalassemia Detection"),
            gr.update(visible=task == "Leukemia Detection")
        )

    task_selector.change(fn=show_section, inputs=task_selector, outputs=[
        clot_section, group_section, sickle_section, thal_section, leukemia_section
    ])

demo.launch(share=True)