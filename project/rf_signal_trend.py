import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Config
LOOKBACK = 5
PREDICT_FORWARD = 5
SMOOTH_WINDOW = 3
OUTPUT_DIR = "/home/cheng-sian/project/signal/results_trend_new" # change the path
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS_NUMERIC = ['signal', 'dist_1', 'dist_2', 'signal_rollmean3', 'signal_diff', 'signal_rollstd3']
FEATURE_COLS_CATEGORICAL = ['antenna', 'scenario']

# Load and filter data
df = pd.read_csv("/home/cheng-sian/project/avg_losses/drl/dataset_for_drl.csv") # change the path
df = df[
    (df['bandwidth'] == 80) &
    (df['users'] == 'n1') &
    (df['duplication'] == 'nodup') &
    (df['fps'] == '1fps') &
    (df['deadline'] == 0.02)
].copy()

group_cols = ['users', 'fps', 'antenna', 'scenario', 'duplication', 'deadline', 'bandwidth']

# Feature Engineering
df['signal_rollmean3'] = df.groupby(group_cols)['signal'].transform(lambda x: x.rolling(3).mean())
df['signal_rollstd3'] = df.groupby(group_cols)['signal'].transform(lambda x: x.rolling(3).std())
df['signal_diff'] = df.groupby(group_cols)['signal'].diff()
df['signal_avg_now'] = df.groupby(group_cols)['signal'].transform(lambda x: x.rolling(SMOOTH_WINDOW).mean())
df['signal_avg_next'] = df.groupby(group_cols)['signal'].shift(-SMOOTH_WINDOW+1).rolling(SMOOTH_WINDOW).mean()
df['slope_smooth'] = df['signal_avg_next'] - df['signal_avg_now']
df = df.dropna(subset=['signal_rollmean3', 'signal_avg_now', 'signal_avg_next']).reset_index(drop=True)

# One-hot encoding for categorical features
encoder = OneHotEncoder(sparse_output=False)
df_cat = encoder.fit_transform(df[FEATURE_COLS_CATEGORICAL])
df_cat_df = pd.DataFrame(df_cat, columns=encoder.get_feature_names_out(FEATURE_COLS_CATEGORICAL))
df = pd.concat([df.reset_index(drop=True), df_cat_df.reset_index(drop=True)], axis=1)

# Target Label: slope_class
def classify_slope(val, tol=0.5):
    if val > tol:
        return "up"
    elif val < -tol:
        return "down"
    else:
        return "flat"

df['slope_class'] = df['slope_smooth'].apply(classify_slope)
df = df.sort_values(by=group_cols + ['time']).reset_index(drop=True)

# Prepare Input-Output Samples
FEATURE_COLS_ONEHOT = FEATURE_COLS_NUMERIC + list(df_cat_df.columns)
samples = []
targets = {f't{i}_class': [] for i in range(1, PREDICT_FORWARD + 1)}

for _, group_df in df.groupby(group_cols):
    group_df = group_df.reset_index(drop=True)
    for i in range(len(group_df) - LOOKBACK - PREDICT_FORWARD + 1):
        x_seq = group_df.loc[i:i+LOOKBACK-1, FEATURE_COLS_ONEHOT].values.flatten()
        samples.append(x_seq)
        for j in range(PREDICT_FORWARD):
            targets[f't{j+1}_class'].append(group_df.loc[i+LOOKBACK+j, 'slope_class'])

X = np.array(samples)
results = pd.DataFrame()

# Train + Predict per time step
for t in range(1, PREDICT_FORWARD + 1):
    y = targets[f't{t}_class']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_probas, all_trues = [], [], []

    for train_idx, test_idx in skf.split(X, y_enc):
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf.fit(X[train_idx], y_enc[train_idx])
        y_pred = clf.predict(X[test_idx])
        y_proba = clf.predict_proba(X[test_idx])

        all_preds.extend(y_pred)
        all_probas.extend(y_proba)
        all_trues.extend(y_enc[test_idx])

    pred_classes = le.inverse_transform(all_preds)
    true_classes = le.inverse_transform(all_trues)
    confidences = np.array(all_probas)[np.arange(len(all_preds)), all_preds]

    results[f"true_class_t{t}"] = true_classes
    results[f"pred_class_t{t}"] = pred_classes
    results[f"conf_t{t}"] = confidences

    # Save model and encoder
    joblib.dump(clf, f"{OUTPUT_DIR}/rf_classifier_t{t}.joblib")
    joblib.dump(le, f"{OUTPUT_DIR}/label_encoder_t{t}.joblib")

# Save Results
results.to_csv(f"{OUTPUT_DIR}/signal_trend_predictions.csv", index=False)
print("Prediction results and confidences saved.")
