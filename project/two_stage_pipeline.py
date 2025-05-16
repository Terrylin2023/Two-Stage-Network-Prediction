import os
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay

def load_and_preprocess_data(data_path, bandwidth):
    df = pd.read_csv(data_path)
    original_features = ['users', 'fps', 'antenna', 'scenario', 'duplication', 'deadline', 'time', 'bandwidth', 'losses', 'dist_1', 'dist_2', 'signal']
    df = df[original_features].dropna().reset_index(drop=True)
    df = df[df['bandwidth'] == bandwidth].copy()
    df = df[(df['losses'] >= 1) & (df['losses'] <= 40)].copy()

    group_cols = ['users', 'fps', 'antenna', 'scenario', 'duplication', 'deadline']
    df = df.sort_values(by=group_cols + ['time']).reset_index(drop=True)

    df['signal_next'] = df.groupby(group_cols + ['bandwidth'], group_keys=False)['signal'].shift(-1)
    df['loss_th_25_next'] = df.groupby(group_cols + ['bandwidth'], group_keys=False)['losses'].shift(-1) > 25
    df['loss_th_25_next'] = df['loss_th_25_next'].astype(int)
    df = df.dropna().reset_index(drop=True)

    return df

def train_signal_model(train_df, features, target):
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), features)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42))
    ])
    pipeline.fit(train_df[features], train_df[target])
    return pipeline

def train_loss_model(train_df, model_type='rf'):
    features = ['dist_1', 'dist_2', 'predicted_signal']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), features)
    ])
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', max_depth=20, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(train_df[features], train_df['loss_th_25_next'])
    return pipeline

def evaluate_signal_model(model, test_df):
    predictions = model.predict(test_df[['dist_1', 'dist_2', 'signal']])
    mae = mean_absolute_error(test_df['signal_next'], predictions)
    mse = mean_squared_error(test_df['signal_next'], predictions)
    mape = np.mean(np.abs((test_df['signal_next'] - predictions) / test_df['signal_next'])) * 100
    return mae, mse, mape, predictions

def evaluate_loss_model(model, test_df, name, output_dir):
    preds = model.predict(test_df[['dist_1', 'dist_2', 'predicted_signal']])
    proba = model.predict_proba(test_df[['dist_1', 'dist_2', 'predicted_signal']])[:, 1]
    acc = accuracy_score(test_df['loss_th_25_next'], preds)
    report = classification_report(test_df['loss_th_25_next'], preds)
    auc = roc_auc_score(test_df['loss_th_25_next'], proba)
    RocCurveDisplay.from_predictions(test_df['loss_th_25_next'], proba)
    plt.title(f"{name} ROC (AUC={auc:.3f})")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"roc_curve_{name.lower().replace(' ', '_')}.png"))
    plt.close()
    return acc, report, auc, preds, proba

def save_outputs(output_dir, signal_model, loss_model, logistic_model, test_df, results):
    joblib.dump(signal_model, os.path.join(output_dir, "random_forest_signal.pkl"))
    joblib.dump(loss_model, os.path.join(output_dir, "random_forest_loss.pkl"))
    joblib.dump(logistic_model, os.path.join(output_dir, "logistic_regression_loss.pkl"))

    pd.DataFrame(results).to_csv(os.path.join(output_dir, "model_comparison_metrics.csv"), index=False)

    print("Models and metrics saved successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs')
    parser.add_argument('--bandwidth', type=int, default=80, help='Bandwidth filter')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_and_preprocess_data(args.data_path, args.bandwidth)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['loss_th_25_next'], random_state=42)

    # First model
    signal_model = train_signal_model(train_df, ['dist_1', 'dist_2', 'signal'], 'signal_next')
    mae, mse, mape, pred_signal_test = evaluate_signal_model(signal_model, test_df)
    print(f"Signal Prediction - MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%")

    train_df['predicted_signal'] = signal_model.predict(train_df[['dist_1', 'dist_2', 'signal']])
    test_df['predicted_signal'] = pred_signal_test

    # Second model - Logistic
    logistic_model = train_loss_model(train_df, model_type='logistic')
    acc_log, report_log, auc_log, preds_log, prob_log = evaluate_loss_model(logistic_model, test_df, 'Logistic Regression', args.output_dir)

    # Second model - RF
    rf_model = train_loss_model(train_df, model_type='rf')
    acc_rf, report_rf, auc_rf, preds_rf, prob_rf = evaluate_loss_model(rf_model, test_df, 'Random Forest', args.output_dir)

    results = {
        'Model': ['Logistic Regression', 'Random Forest'],
        'Accuracy': [acc_log, acc_rf],
        'AUC': [auc_log, auc_rf]
    }

    save_outputs(args.output_dir, signal_model, rf_model, logistic_model, test_df, results)

if __name__ == '__main__':
    main()
