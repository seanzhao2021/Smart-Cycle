# Experiments

This document serves as the technical deep dive into the Smart-Waste project. Highlighting testing procedures and methods and providing insight into why the final approach was chosen.

## Dataset
i initially tried my own combiation of taco and openwaste
but had troubles not enough labeling data

eventually used dwaste dataset

## Models
### Classification Models

| Model Name | Loss | Accuracy | Precision | Recall | F1 Macro |
| ---------- | ---- | -------- | --------- | ------ | -------- |
| MobileNet | 0.164 | 0.941 | 0.935 | 0.937 | 0.935 |
| ResNet-50 | 0.128 | 0.958 | 0.955 | 0.959 | 0.957 |
| ResNet-101 | ---- | -------- | --------- | ------ | -------- |
| EfficientNetV2-S | ---- | -------- | --------- | ------ | -------- |
| EfficientNetV2-M | ---- | -------- | --------- | ------ | -------- |



### Detection Models
**YOLOv8n**

**YOLO11n**

## Ablation Studies


## Metrics

| Metric | Significance |
| ----- | ----- |
| Loss | Measures how wrong model's predictions are |
| Accuracy | Correct predictions percentage |
| Precision | Correct predicted positives |
| Recall | Correct positives detected out of all positives |
| F1 Macro | Harmonic mean of precision and recall |
| mAP50 | Mean average precision at IoU = 0.5. A detection is considered correct if overlap is greater than or equal to 50% |
| mAP50-95 | Much stricter measure than mAP50. Average mAP across IoU thresholds from 0.5 to 0.95 |

### Interpretation


## Observations and Conclusions