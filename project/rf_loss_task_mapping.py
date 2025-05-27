import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Config
DATA_PATH = "/home/cheng-sian/project/avg_losses/drl/dataset_for_drl.csv"
OUTPUT_DIR = "/home/cheng-sian/project/prob/results_mul_class_multistep_rf_scenario"
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURE_COLS = ['dist_1', 'dist_2', 'signal', 'antenna','scenario']
GROUP_COLS = ['users', 'fps', 'antenna', 'scenario', 'duplication', 'deadline']
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and preprocess data
df = pd.read_csv(DATA_PATH)
df = df[(df['bandwidth'] == 80) & (df['duplication'] == 'nodup')].dropna().reset_index(drop=True)

# Define bins and labels
bins = [-0.01, 0] + list(range(10, 101, 10))
labels = list(range(11))

for t in range(1, 6):
    df[f'cumulative_losses_t{t}'] = (
        df.groupby(GROUP_COLS)['losses'].transform(lambda x: x.shift(-1).rolling(t).sum())
    )
    df[f'loss_pct_t{t}'] = (df[f'cumulative_losses_t{t}'] / (43 * t)) * 100
    df[f'loss_class_t{t}'] = pd.cut(df[f'loss_pct_t{t}'], bins=bins, labels=labels, right=True).astype("Int64")

# Accuracy mapping values
vgg16_acc =    [0.65, 0.41, 0.25, 0.12, 0.08, 0.06, 0.05, 0.06, 0.01, 0.02, 0.0]
resnet9_acc =  [0.68, 0.63, 0.42, 0.13, 0.10, 0.07, 0.06, 0.05, 0.02, 0.03, 0.0]
yolov3_map =   [0.60, 0.58, 0.52, 0.45, 0.32, 0.23, 0.14, 0.09, 0.03, 0.01, 0.0]
yolov8_map =   [0.66, 0.60, 0.52, 0.46, 0.31, 0.22, 0.13, 0.08, 0.03, 0.01, 0.0]

# Traiining
for t in range(1, 6):
    print(f"Training model for T+{t}s")
    target_col = f'loss_class_t{t}'
    df_t = df.dropna(subset=[target_col]).copy()
    train_df, test_df = train_test_split(df_t, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)

    le = LabelEncoder()
    train_df['loss_class_encoded'] = le.fit_transform(train_df[target_col])
    test_df['loss_class_encoded'] = le.transform(test_df[target_col])

    preprocessor = ColumnTransformer([
        ('num', 'passthrough', ['dist_1', 'dist_2', 'signal']),
        ('cat', OneHotEncoder(), ['antenna', 'scenario'])
    ])

    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, class_weight='balanced')
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    pipeline.fit(train_df[FEATURE_COLS], train_df['loss_class_encoded'])

    # Predict
    y_proba = pipeline.predict_proba(test_df[FEATURE_COLS])
    pred_idx = np.argmax(y_proba, axis=1)
    pred_class = le.inverse_transform(pred_idx).astype(int)
    confidence = np.max(y_proba, axis=1)

    # Map to expected accuracy
    mapped_vgg16 = [vgg16_acc[int(cls)] for cls in pred_class]
    mapped_resnet9 = [resnet9_acc[int(cls)] for cls in pred_class]
    mapped_yolov3 = [yolov3_map[int(cls)] for cls in pred_class]
    mapped_yolov8 = [yolov8_map[int(cls)] for cls in pred_class]

    # Save result
    output_df = test_df.reset_index(drop=True).copy()
    output_df[f'Predicted_Loss_Class_T{t}'] = pred_class
    output_df[f'Decision_Confidence_T{t}'] = confidence
    output_df[f'Expected_Accuracy_VGG16_T{t}'] = mapped_vgg16
    output_df[f'Expected_Accuracy_ResNet9_T{t}'] = mapped_resnet9
    output_df[f'Expected_Accuracy_YOLOv3_T{t}'] = mapped_yolov3
    output_df[f'Expected_Accuracy_YOLOv8_T{t}'] = mapped_yolov8

    output_df[[target_col,
               f'Predicted_Loss_Class_T{t}',
               f'Decision_Confidence_T{t}',
               f'Expected_Accuracy_VGG16_T{t}',
               f'Expected_Accuracy_ResNet9_T{t}',
               f'Expected_Accuracy_YOLOv3_T{t}',
               f'Expected_Accuracy_YOLOv8_T{t}']].to_csv(
        os.path.join(OUTPUT_DIR, f"predicted_confidence_t{t}.csv"), index=False
    )

    joblib.dump(pipeline, os.path.join(OUTPUT_DIR, f"loss_model_t{t}.pkl"))
    print(f"[T+{t}s] saved predicted confidence and accuracy mapping file.")
