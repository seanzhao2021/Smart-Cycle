# Smart-Cycle
AI powered smart recycling web app. Documented training and testing procedures in docs/experiments.md. On web app, users can upload images to scan for waste or use a webcam and detect from live feed. After app detects which class the waste belongs to, it will provide users with insight on how to responsibly dispose of it.

# Dependencies

backend
```
cd backend
pip install requirements.txt
```

frontend
```
cd frontend
npm install
```

Dataset can be downloaded at https://www.kaggle.com/datasets/sumn2u/dwaste-data-v4-annotated
Note you will need to split your dataset into train, test, and val split folders if you want to train your own model.

## Training

```
python train_classification.py --data_dir <PATH_TO_DATASET> --model_name <model> --epochs 20 --batch_size 32 --img_size 224 --output_dir runs
```

model - mobilenet, resnet50, resnet101, efficientnetv2_s, efficientnetv2_s

To run detection models.

Baseline Yolo models
```
yolo detect train model=yolov8n.pt data=data.yaml epochs=20 imgsz=640
```

Yolo models integrated with CBAM
make sure to update proper dataset yaml path and model yaml path first
```
python train_cbam_yolo.py
```
Training graphs can be observed in ultralytics/runs/detect/

## Smart Cycle Web App Startup
To start up the web app
```
cd backend
uvicorn main:app --reload
```
in another terminal
```
cd frontend
npm run dev
```
# Key Results

Key results and insights can be found in docs/experiments.md