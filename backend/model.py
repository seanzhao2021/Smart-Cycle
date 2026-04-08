from PIL import Image
from ultralytics import YOLO

model = YOLO(r"C:\Users\seanz\Smart-Cycle-dummy\runs\detect\train2\weights\best.pt")

def predict_image(image: Image.Image):
    width, height = image.size

    # YOLOv8 returns a list (one result per image)
    results = model(image)
    detections = []
    result = results[0]

    for box in result.boxes:
        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Confidence
        conf = float(box.conf[0])

        # Class index → label
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        detections.append({
            "class": cls_name,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2]
        })

    return detections