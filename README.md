# License Plate Detection & OCR

A two-stage pipeline that detects license plates in images and videos, then reads the plate text using OCR.

**Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/coolbambook/OCR_car_plates)

![Demo](https://github.com/user-attachments/assets/e20b956d-9d48-4d8c-bf70-2a3105d2fee4)

---

## Pipeline

```
Image / Video
      │
      ▼
YOLOv8 (fine-tuned) → detects plate location
      │
      ▼
Crop plate region
      │
      ▼
TrOCR (fine-tuned on license plates) → reads plate text
      │
      ▼
Output: bounding box + plate text
```

---

## Models

| Stage | Model | Purpose |
|---|---|---|
| Detection | YOLOv8n (fine-tuned) | Locate license plate in image |
| OCR | TrOCR-base-printed | Read text from cropped plate |

---

## Dataset

[Car Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) — 443 images with Pascal VOC annotations.

- Train: 354 images
- Val: 89 images

---

## Results

| Metric | Score |
|---|---|
| mAP@0.5 | *your score here* |
| mAP@0.5:0.95 | *your score here* |
| Inference speed | ~1.8 FPS |

---

## Limitations

- OCR accuracy varies by plate style, country, and image quality
- ~1.8 FPS due to TrOCR inference time (~500ms per plate). Suitable for low-frequency detection (parking systems, toll gates) rather than real-time video
- For real-time use, replacing TrOCR with a lighter model or ONNX optimization would be needed

---

## Run Locally

```bash
pip install ultralytics transformers pillow opencv-python
```

```python
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

yolo      = YOLO("best.pt")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
trocr     = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Detect plate
img        = Image.open("car.jpg").convert("RGB")
detections = yolo.predict(img, conf=0.3)[0]

# Read text
for box in detections.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cropped         = img.crop((x1, y1, x2, y2))
    pixel_values    = processor(cropped, return_tensors="pt").pixel_values
    generated_ids   = trocr.generate(pixel_values)
    text            = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Plate: {text}")
```

---

## Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Microsoft TrOCR](https://huggingface.co/microsoft/trocr-base-printed)
- [Gradio](https://gradio.app)
