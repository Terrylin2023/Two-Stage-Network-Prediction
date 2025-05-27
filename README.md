# Wireless Network Prediction Pipeline

This repository contains two machine learning pipelines for wireless signal analysis and loss prediction:

1. **Signal Trend Classification** (multi-step)
2. **Loss Class Prediction + Object Detection Accuracy Mapping**

Both models are built using Random Forests and applied on preprocessed wireless network datasets.

---

## Signal Trend Classification

###  Task Description
Predict the **trend** of signal strength (up / down / flat) for the next 5 seconds based on past measurements and device configuration.

###  Model
- Classifier: `RandomForestClassifier`
- Prediction Targets: `slope_class` at T+1, T+2, ..., T+5
- Input Features:
  - Numerical: `signal`, `dist_1`, `dist_2`, `signal_rollmean3`, `signal_diff`, `signal_rollstd3`
  - Categorical: `antenna`, `scenario` (one-hot encoded)

###  Output

Each file:  
 `signal_trend_predictions.csv`

| true_class_t1 | pred_class_t1 | conf_t1 | ... |
|---------------|----------------|---------|-----|
| down          | down           | 0.83    |     |

- `true_class_t{i}`: Ground truth trend class
- `pred_class_t{i}`: Model-predicted class
- `conf_t{i}`: Confidence of prediction (probability of predicted class)

---

##  Loss Class Prediction + Accuracy Mapping

###  Task Description
Predict the **loss class (0%~100%)** for the next 5 seconds, then map each class to the **expected object detection accuracy** under different model architectures (VGG16, ResNet9, YOLOv3, YOLOv8).

###  Model
- Classifier: `RandomForestClassifier`
- Input Features:
  - `dist_1`, `dist_2`, `signal`, `antenna`, `scenario`
- Loss Classes:  
  - `0`: 0%
  - `1`: 0–10%
  - ...
  - `10`: 90–100%

### 📁 Output

Each file:  
 `predicted_confidence_t{i}.csv`

| loss_class_t1 | Predicted_Loss_Class_T1 | Decision_Confidence_T1 | Expected_Accuracy_VGG16_T1 | ... |
|---------------|--------------------------|--------------------------|------------------------------|-----|
| 2             | 2                        | 0.743                    | 0.25                         |     |

- `loss_class_t{i}`: Ground truth loss class
- `Predicted_Loss_Class_T{i}`: Model-predicted loss class
- `Decision_Confidence_T{i}`: Confidence score (max probability)
- `Expected_Accuracy_*_T{i}`: Mapped accuracy based on predicted loss for:
  - VGG16
  - ResNet9
  - YOLOv3
  - YOLOv8

---


