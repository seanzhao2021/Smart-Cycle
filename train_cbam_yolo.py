import sys
sys.path.insert(0, r"<PATH_TO_REPO>\ultralytics")

from ultralytics import YOLO


def main():
    #change to other yaml files for different cbam configs
    yaml_path = r"<PATH_TO_REPO>\yaml\yolov8n_cbam_deep.yaml"
    data_path = r"<PATH_TO_DATASET>"

    model = YOLO(yaml_path)
    model = model.load("yolov8n.pt")

    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name="yolov8n_cbam_deep",
    )


if __name__ == "__main__":
    main()