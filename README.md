# Smart-Cycle
AI powered smart recycling web app.

# Dependencies

pip install ultralytics

# Run

python train_classification.py --data_dir <PATH_TO_DATASET> --model_name <model> --epochs 20 --batch_size 32 --img_size 224 --output_dir runs

model - mobilenet, resnet50, resnet101, efficientnetv2_s, efficientnetv2_s

To run detection models.

Baseline Yolo models
yolo detect train model=yolov8n.pt data=data.yaml epochs=20 imgsz=640

Yolo models integrated with CBAM
make sure to update proper dataset yaml path and model yaml path first
python train_cbam_yolo.py

Training graphs can be observed in ultralytics/runs/detect/

# Key Results

Dataset can be downloaded at https://www.kaggle.com/datasets/sumn2u/dwaste-data-v4-annotated