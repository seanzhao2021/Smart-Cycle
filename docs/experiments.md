# Experiments

This document serves as the technical deep dive into the Smart-Waste project. Highlighting testing procedures and methods and providing insight into why the final approach was chosen.

## Dataset
i initially tried my own combination of taco and openwaste datasets
but had troubles with not enough labeling data for detection models

eventually used dwaste dataset

## Models
### Classification Models

| Model Name | Loss | Accuracy | Precision | Recall | F1 Macro |
| ---------- | ---- | -------- | --------- | ------ | -------- |
| MobileNet | 0.164 | 0.941 | 0.935 | 0.937 | 0.935 |
| ResNet-50 | 0.128 | 0.958 | 0.955 | 0.959 | 0.957 |
| ResNet-101 | 0.108 | 0.966 | 0.965 | 0.966 | 0.965 |
| EfficientNetV2-S | 0.307 | 0.932 | 0.929 | 0.931 | 0.929 |
| EfficientNetV2-M | 0. | 0. | 0. | 0. | 0. |



### Detection Models

| Model Name | Epochs | Precision | Recall | mAP50 | map50-95 |
| ---------- | ------ | --------- | ------ | ----- | -------- |
| YOLOv8n | 20 | 0.867 | 0.814 | 0.89 | 0.669 |
| YOLO11n | 20 | 0.886 | 0.806 | 0.891 | 0.665 |
| YOLOv8n (Single CBAM after SPPF) | 20 | 0.811 | 0.821 | 0.879 | 0.647 |
| YOLOv8n (Multi CBAM, all backbone stages) | 20 | 0.828 | 0.736 | 0.84 | 0.616 |
| YOLOv8n (Multi CBAM, all backbone stages) | 50 | 0.828 | 0.808 | 0.877 | 0.652 |
| YOLOv8n (Deep backbone CBAM) | 20 | 0.817 | 0.784 | 0.865 | 0.637 |


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