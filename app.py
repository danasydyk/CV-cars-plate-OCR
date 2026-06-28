import gradio as gr
import cv2
import numpy as np
import tempfile
import os
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ── Load models ───────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"

yolo   = YOLO("best.pt")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
trocr     = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)

# ── OCR ───────────────────────────────────────────────────────────────────────

def read_plate_text(crop: Image.Image) -> str:
    pixel_values = processor(crop, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = trocr.generate(pixel_values)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ── Core: annotate a single frame ────────────────────────────────────────────

def annotate_frame(frame_bgr: np.ndarray, conf: float = 0.3):
    """Returns annotated BGR frame + list of plate texts."""
    results = yolo.predict(frame_bgr, conf=conf, verbose=False)[0]
    texts   = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf_score      = float(box.conf[0])

        crop       = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).crop((x1, y1, x2, y2))
        plate_text = read_plate_text(crop)
        texts.append(plate_text)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{plate_text} ({conf_score:.2f})"
        cv2.putText(frame_bgr, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame_bgr, texts

# ── Image handler ─────────────────────────────────────────────────────────────

def detect_image(image: Image.Image, conf: float = 0.3):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotated, texts = annotate_frame(frame, conf)
    annotated_rgb    = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    label = "\n".join(texts) if texts else "No license plate detected."
    return annotated_rgb, label

# ── Video handler ─────────────────────────────────────────────────────────────

def detect_video(video_path: str, conf: float = 0.3):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(tempfile.gettempdir(), "plate_output.mp4")
    out      = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    all_texts = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, texts = annotate_frame(frame, conf)
        all_texts.update(texts)
        out.write(annotated)

    cap.release()
    out.release()
    return out_path, "\n".join(all_texts) if all_texts else "No plates detected."

# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="License Plate Detector") as demo:
    gr.Markdown("# 🚗 License Plate Detector")
    gr.Markdown("Upload an **image** or **video** to detect and read license plates using YOLOv8 + TrOCR.")

    conf_slider = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Confidence Threshold")

    with gr.Tab("Image"):
        with gr.Row():
            img_input  = gr.Image(type="pil", label="Upload Image")
            img_output = gr.Image(type="numpy", label="Detected Plates")
        txt_output = gr.Textbox(label="Plate Text")
        gr.Button("Detect").click(detect_image,
                                  inputs=[img_input, conf_slider],
                                  outputs=[img_output, txt_output])

    with gr.Tab("Video"):
        vid_input  = gr.Video(label="Upload Video")
        vid_output = gr.Video(label="Annotated Video")
        vid_texts  = gr.Textbox(label="All Detected Plates")
        gr.Button("Detect").click(detect_video,
                                  inputs=[vid_input, conf_slider],
                                  outputs=[vid_output, vid_texts])

demo.launch(allowed_paths=[tempfile.gettempdir()])
