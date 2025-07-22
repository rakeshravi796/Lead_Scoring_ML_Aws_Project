from flask import Flask, request, render_template
import mlflow
import mlflow.pyfunc
import pandas as pd
import pickle
import os
import boto3
import sagemaker
import json
import re
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce

app = Flask(__name__)

# S3 Bucket and Input Path
bucket = '**************'
input_s3_path = '**********/train_input_test/Lead_Scoring_training.csv'

# === Load MLflow Model from SageMaker ===
region = boto3.Session().region_name
role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.Session()

mlflow.set_tracking_uri(
    "***********:mlflow-tracking-server/final-project-tracking-server"
)
MODEL_NAME = "LeadScoringBestTunedModel"
client = mlflow.tracking.MlflowClient()
latest_versions = client.get_latest_versions(MODEL_NAME)
latest_version = max(latest_versions, key=lambda v: int(v.version))
model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
model = mlflow.pyfunc.load_model(model_uri)

# === Fallback to Local Pickled Model (Optional) ===
try:
    with open('Pickles/model_new.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Warning: Could not load local model. Using MLflow model instead. {e}")

# === Load Cleaning Pipeline ===
PIPELINE_PATH = "Pickles/lead_scoring_pipeline.pkl"
if not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError(f"⚠ Pipeline file not found at {PIPELINE_PATH}")

with open(PIPELINE_PATH, "rb") as f:
    cleaning_pipeline = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part in the request"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    try:
        # Load uploaded data
        df = pd.read_csv(file)

        # Clean the data before prediction
        df_clean = cleaning_pipeline.transform(df)

        # Predict using the model
        predictions = model.predict(df_clean)

        # Add predictions to original dataframe
        df['Predicted_Salary'] = predictions

        # Convert results to HTML
        table_html = df.to_html(index=False, classes="table table-bordered")

        # Load reference data for drift analysis
        reference_df = pd.read_csv(input_s3_path)
        reference_clean = reference_df.drop(["Converted"], axis=1)

        # Drift report
        def sanitize_name(name):
            return re.sub(r"[^\w\-/ .]", "_", name)

        report = Report([DataDriftPreset()])
        report.run(reference_clean, df_clean)
        report_data = json.loads(report.json())

        # Log drift metrics to MLflow
        mlflow.set_experiment("evidently_hist_vs_new")
        with mlflow.start_run():
            for metric in report_data.get("metrics", []):
                metric_id = metric.get("metric_id", "")
                value = metric.get("value", None)
                if metric_id.startswith("Drifted") and isinstance(value, dict):
                    mlflow.log_metric("Number_of_drifted_columns", value.get('count', 0))
                else:
                    clean_metric_id = sanitize_name(metric_id.lower().replace(" ", "_"))
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(clean_metric_id, value)

        return render_template('index.html', table=table_html)

    except Exception as e:
        return f"❌ An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
