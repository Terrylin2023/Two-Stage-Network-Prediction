import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

#  Constants 
DATA_PATH = "/home/cheng-sian/project/avg_losses/drl/dataset_for_drl.csv"
OUTPUT_DIR = "/home/cheng-sian/project/prob/results_mul_class_multistep_rf"
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURE_COLS_LOSS = ['dist_1', 'dist_2', 'signal', 'duplication', 'antenna']
GROUP_COLS = ['users', 'fps', 'antenna', 'scenario', 'duplication', 'deadline']
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  Load and preprocess data 
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df[df['bandwidth'] == 80]
df = df[df['duplication'] == 'nodup']
df = df.dropna().reset_index(drop=True)

# Define loss bins and labels
bins = [0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = list(range(10))
for t in range(1, 6):
    df[f'cumulative_losses_t{t}'] = (
        df.groupby(GROUP_COLS)['losses']
          .transform(lambda x: x.shift(-1).rolling(t).sum())
    )
    denominator = 43 * t
    df[f'loss_pct_t{t}'] = (df[f'cumulative_losses_t{t}'] / denominator) * 100
    df[f'loss_class_t{t}'] = pd.cut(df[f'loss_pct_t{t}'], bins=bins, labels=labels, right=False).astype("Int64")

#  Multi-step classification 
results = []

for t in range(1, 6):
    print(f"\n=== Training for T+{t}s ===")
    target_col = f'loss_class_t{t}'

    df_t = df.dropna(subset=[target_col]).copy()
    train_df, test_df = train_test_split(df_t, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)

    # Encode labels
    le = LabelEncoder()
    train_df['loss_class_encoded'] = le.fit_transform(train_df[target_col])
    test_df['loss_class_encoded'] = le.transform(test_df[target_col])
    encoded_classes = np.arange(len(le.classes_))
    original_classes = le.inverse_transform(encoded_classes)

    # Preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['dist_1', 'dist_2', 'signal']),
        ('cat', OneHotEncoder(), ['duplication', 'antenna'])
    ])

    # Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced')
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    pipeline.fit(train_df[FEATURE_COLS_LOSS], train_df['loss_class_encoded'])

    # Predict & Evaluate
    y_pred = pipeline.predict(test_df[FEATURE_COLS_LOSS])
    y_true = test_df['loss_class_encoded']
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=le.transform(original_classes), target_names=[str(int(c)) for c in original_classes],output_dict=True)
    results.append((t, acc))
    
    n_classes = len(np.unique(y_true))
    print(f"[T+{t}s] Number of classes: {n_classes}")
    original_classes = le.classes_
    # ROC Curve
    y_proba = pipeline.predict_proba(test_df[FEATURE_COLS_LOSS])
    lb = LabelBinarizer()
    lb.fit(encoded_classes)
    y_true_bin = lb.transform(y_true)
    if n_classes < 2:
        print(f"[T+{t}s] Only {n_classes} class present. Skipping AUC.")
        macro_auc = np.nan
    elif n_classes == 2:
        macro_auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        y_true_bin = LabelBinarizer().fit_transform(y_true)
        macro_auc = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, cls in enumerate(original_classes):
        if n_classes == 2:
            fpr[cls], tpr[cls], _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc[cls] = auc(fpr[cls], tpr[cls])
            break  # Only one curve needed
        else:
            fpr[cls], tpr[cls], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc[cls] = auc(fpr[cls], tpr[cls])

    # Plot ROC
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for idx, cls in enumerate(original_classes):
        if cls in fpr:
            plt.plot(fpr[cls], tpr[cls], color=colors[idx % 10],
                    label=f"Class {int(cls)} (AUC = {roc_auc[cls]:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"T+{t}s - ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"roc_curve_t{t}.png"))
    plt.close()

    # Save classification report
    pd.DataFrame(report).transpose().to_csv(os.path.join(OUTPUT_DIR, f"classification_report_t{t}.csv"))
    #  Map to expected model accuracy (VGG16 & ResNet9) 
    vgg16_acc =    [0.65, 0.41, 0.25, 0.12, 0.08, 0.06, 0.05, 0.06, 0.01, 0.02]
    resnet9_acc =  [0.68, 0.63, 0.42, 0.13, 0.10, 0.07, 0.06, 0.05, 0.02, 0.03]
    yolov3_map = [0.60, 0.58, 0.52, 0.45, 0.32, 0.23, 0.14, 0.09, 0.03, 0.01]
    yolov8_map = [0.66, 0.60, 0.52, 0.46, 0.31, 0.22, 0.13, 0.08, 0.03, 0.01]

    predicted_class_idx = np.argmax(y_proba, axis=1)
    predicted_class = le.inverse_transform(predicted_class_idx)
    predicted_class = predicted_class.astype(int)

    mapped_vgg16 = [vgg16_acc[int(cls)] for cls in predicted_class]
    mapped_resnet9 = [resnet9_acc[int(cls)] for cls in predicted_class]
    mapped_yolov3 = [yolov3_map[int(cls)] for cls in predicted_class]
    mapped_yolov8 = [yolov8_map[int(cls)] for cls in predicted_class]

    output_df = test_df.reset_index(drop=True).copy()
    output_df[f'Predicted_Loss_Class_T{t}'] = predicted_class
    output_df[f'Expected_Accuracy_VGG16_T{t}'] = mapped_vgg16
    output_df[f'Expected_Accuracy_ResNet9_T{t}'] = mapped_resnet9
    output_df[f'Expected_Accuracy_YOLOv3_T{t}'] = mapped_yolov3
    output_df[f'Expected_Accuracy_YOLOv8_T{t}'] = mapped_yolov8

    output_path = os.path.join(OUTPUT_DIR, f"predicted_accuracy_t{t}.csv")
    output_df[[target_col, 
                f'Predicted_Loss_Class_T{t}',
                f'Expected_Accuracy_VGG16_T{t}',
                f'Expected_Accuracy_ResNet9_T{t}',
                f'Expected_Accuracy_YOLOv3_T{t}',
                f'Expected_Accuracy_YOLOv8_T{t}']].to_csv(output_path, index=False)
    print(f"[T+{t}s] Predicted accuracy mapping saved to {output_path}")
    # Save model
    import joblib
    joblib.dump(pipeline, os.path.join(OUTPUT_DIR, f"loss_model_t{t}.pkl"))

# Save accuracy results
results_df = pd.DataFrame(results, columns=["Time Ahead (s)", "Accuracy"])
results_df.to_csv(os.path.join(OUTPUT_DIR, "multi_step_accuracy.csv"), index=False)

# Plot accuracy trend
plt.figure(figsize=(8, 5))
plt.plot(results_df["Time Ahead (s)"], results_df["Accuracy"], marker='o')
plt.xlabel("Prediction Horizon (Seconds Ahead)")
plt.ylabel("Accuracy")
plt.title("Multi-step Loss Classification Accuracy")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "multi_step_accuracy_plot.png"))
plt.close()

for t in range(1, 6):
    pred_df = pd.read_csv(os.path.join(OUTPUT_DIR, f"predicted_accuracy_t{t}.csv"))
    pred_col = f'Predicted_Loss_Class_T{t}'
    class_counts = pred_df[pred_col].value_counts().sort_index()
    class_counts.index = class_counts.index.astype(int)
    plt.figure(figsize=(8, 4))
    class_counts.plot(kind='bar')
    plt.title(f"Class Distribution for T+{t}s (Predicted)")
    plt.xlabel("Loss Class")
    plt.ylabel("Sample Count")
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"class_distribution_t{t}.png"))
    plt.close()

class_dist_all = []
for t in range(1, 6):
    pred_df = pd.read_csv(os.path.join(OUTPUT_DIR, f"predicted_accuracy_t{t}.csv"))
    pred_col = f'Predicted_Loss_Class_T{t}'
    counts = pred_df[pred_col].value_counts().sort_index()
    counts.index = counts.index.astype(int)
    counts.name = f"T+{t}s"
    class_dist_all.append(counts)
dist_df = pd.concat(class_dist_all, axis=1).fillna(0).astype(int).sort_index()
plt.figure(figsize=(10, 6))
for cls in dist_df.index:
    plt.plot(dist_df.columns, dist_df.loc[cls], marker='o', label=f"Class {cls}")
plt.xlabel("Time Horizon")
plt.ylabel("Sample Count")
plt.title("Loss Class Trend Over Time")
plt.legend(title="Class")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_class_trend.png"))
plt.close()
