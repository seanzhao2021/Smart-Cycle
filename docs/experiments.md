# Experiments
This document serves as the technical dive into the Smart-Waste project. Highlighting testing procedures and methods and providing insight into why the final approach was chosen.

## Dataset
Initial Combined dataset (TACO + OpenWaste) ran into problems due to limited bounding box annotations
Detection models require significantly more labeled data than classification models

DWaste dataset was selected as final dataset due to
-Nearly 20,000 labeled bounding box instances
-Better distribution amongst 7 classes
-Better suitability for detection model training

## Models
### Classification Models

| Model Name | Loss | Accuracy | Precision | Recall | F1 Macro |
| ---------- | ---- | -------- | --------- | ------ | -------- |
| MobileNet | 0.164 | 0.941 | 0.935 | 0.937 | 0.935 |
| ResNet-50 | 0.128 | 0.958 | 0.955 | 0.959 | 0.957 |
| ResNet-101 | 0.108 | 0.966 | 0.965 | 0.966 | 0.965 |
| EfficientNetV2-S | 0.307 | 0.932 | 0.929 | 0.931 | 0.929 |
| EfficientNetV2-M | 0.342 | 0.927 | 0.923 | 0.928 | 0.925 |

**Conclusion**
Across the five classification architectures tested, ResNet-101 performed the best overall, with ResNet-50 performing the second best. ResNet-101 had the highest recorded performance across all metrics with the greatest generalization. ResNet-50 performed nearly as well as ResNet-101, demonstrating that it is a strong practical alternative. Another point of discussion is in MobileNet's performance, where accuracy and F1 macro were only slightly lower than the ResNet family. However MobileNet is a much lighter model compared to ResNet which could make it ideal for mobile or edge deployment. Finally the EfficientNetV2 models underperformed compared to the other models.

### Detection Models
For object detection, I evaluated multiple YOLO-based architectures, including baseline models and CBAM enhanced variations across 20-50 epochs. Performance was measured with precision, recall, and mAP scores.

| Model Name | Epochs | Precision | Recall | mAP50 | map50-95 |
| ---------- | ------ | --------- | ------ | ----- | -------- |
| YOLOv8n | 20 | 0.867 | 0.814 | 0.89 | 0.669 |
| YOLO11n | 20 | 0.886 | 0.806 | 0.891 | 0.665 |
| YOLOv8n (Single CBAM after SPPF) | 20 | 0.811 | 0.821 | 0.879 | 0.647 |
| YOLOv8n (Multi CBAM, all backbone stages) | 20 | 0.828 | 0.736 | 0.84 | 0.616 |
| YOLOv8n (Multi CBAM, all backbone stages) | 50 | 0.828 | 0.808 | 0.877 | 0.652 |
| YOLOv8n (Deep backbone CBAM) | 20 | 0.817 | 0.784 | 0.865 | 0.637 |
| YOLOv8n (Deep backbone CBAM) | 50 | 0.876 | 0.802 | 0.887 | 0.66 |

**Conclusion**
Baseline models (YOLOv8n and YOLO11n) outperformed all CBAM enhanced variations across most to all metrics. YOLOv8n provided slightly better localization (mAP50-95 scored higher) while YOLO11n provided slightly better precision.

Single CBAM after SPFF noticed a minimal drop in performance against baseline models.
Multi CBAM variant had lower mAP scores across the board and a noticeable drop in recall. Increasing epochs to 50 improved performance but models still did not surpass baseline models.
Deep backbone CBAM noticed moderate performance decrease with no clear advantage over baseline models.

Adding attention modules did not improve detection performance and often decreased it. Furthermore, increased architectural complexity may introduce optimization difficulty. Baseline models demonstrated high mAP50 scores, showcasing strong detection capability with lower mAP50-95 scores, indicating slight weakness in bounding box precision.

## Ablation Studies
**CBAM Integration/Placement**
I evaluated the impact of integrating convolutional block attention modules(CBAM) at different locations within the YOLOv8 architecture (single layer, deep backbone, multi-stage configurations). The goal was to understand how attention modules could affect feature extraction vs aggregation. Datasets, model size, and hyperparameters were kept the same, with only CBAM placement being varied.

| Configuration | Precision | Recall | mAP50 | mAP50-95 |
|--------------|----------|--------|------|----------|
| Baseline YOLOv8n | 0.867 | 0.814 | 0.89 | 0.669 |
| Single CBAM | 0.811 | 0.821 | 0.879 | 0.647 |
| Multi CBAM | 0.828 | 0.736 | 0.84 | 0.616 |
| Deep CBAM | 0.876 | 0.802 | 0.887 | 0.66 |

### Findings

-CBAM failed to meaningfully strengthen model performance compared to baseline
-CBAM increased recall, but consistently reduced precision
-Multi-Stage CBAM significantly reduced perforamnce, likely from over-complexity
-Lightweight YOLO architectures may not benefit from attention modules as they already have optimized feature extraction pipelines

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

-High mAP50 across both baseline detection models indicates strong detection capability --> model can identify waste reliably
-Lower mAP50-95 indicates that bounding box localization is less precise, especially true for small/irregularly shaped objects
-CBAM integration showed slightly higher recall at times but lower overall precision, indicating that those specific attention module integrations made the YOLO model more aggressive in object detection
-Precision drop observed from CBAM integration suggests that attention layers may disrupt learned feature hierarchies or introduce noise

## Observations and Conclusions
**Tradeoffs for consideration**
-Larger classification models (ResNet-101) achieved higher accuracy though they are less suitable for edge deployment
-MobileNet provides a strong accuracy-efficiency tradeoff for real-time applications
-CBAM increases model complexity but did not yield performance gains in this setup
-Detection models prioritize localization accuracy (mAP) over raw classification accuracy

**Conclusions**
-ResNet-101 achieved the best classification performance, while MobileNet offers a strong lightweight alternative for deployment
-YOLOv8n provided the best balance of precision and localization for detection tasks
-CBAM integration did not improve performance and often degraded precision
-Detection performance is limited more by localization quality (mAP50-95) than detection capability (mAP50)
-Dataset quality and size had a significant impact on detection performance, reinforcing the importance of labeled data

