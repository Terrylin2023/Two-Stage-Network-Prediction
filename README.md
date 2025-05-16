# Two-Stage Signal and Loss Prediction

This project implements a two-stage machine learning pipeline to:
1. Predict future signal strength.
2. Use the predicted signal to estimate the probability that packet loss exceeds a threshold.

## Structure
- `two_stage_pipeline.py`: Main script to run the full pipeline.
- `requirements.txt`: Python dependencies.
- `outputs/`: Where models, plots, and metrics are saved.

## Usage

python two_stage_pipeline.py --data_path /Dataset/dataset_for_drl/dataset_for_drl.csv --output_dir outputs

