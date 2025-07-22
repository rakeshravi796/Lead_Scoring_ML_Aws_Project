
from flask import Flask, request, render_template
import mlflow
import mlflow.pyfunc
import pandas as pd
import pickle
import os
from evidently import Report
from evidently.presets import DataDriftPreset
import json
import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce

app = Flask(__name__)

# === Load MLflow Model ===
MODEL_NAME = "BestLeadScoringModel"
client = mlflow.tracking.MlflowClient()
latest_versions = client.get_latest_versions(MODEL_NAME)
latest_version = max(latest_versions, key=lambda v: int(v.version))
model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
model = mlflow.pyfunc.load_model(model_uri)

# with open('Pickles/model_new.pkl', 'rb') as file:
#     model = pickle.load(file)

# === Load Cleaning Pipeline ===
PIPELINE_PATH = "Flask Website/Pickles/lead_scoring_pipeline.pkl"


if not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError(f"⚠️ Pipeline file not found at {PIPELINE_PATH}")

with open(PIPELINE_PATH, "rb") as f:
    cleaning_pipeline = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"

    try:
        df = pd.read_csv(file)

        # Clean the data before prediction
        df_clean = cleaning_pipeline.transform(df)

        # Predict using the model
        predictions = model.predict(df_clean)

        # Add predictions to original df (can choose cleaned df too)
        df['Predicted_Salary'] = predictions

        # Convert to HTML table
        table_html = df.to_html(index=False, classes="table table-bordered")

        # Load reference data
        reference_df = pd.read_csv("./Datasets/Lead Scoring.csv")
        reference_clean = cleaning_pipeline.transform(reference_df)

        #Function to sanitize the names for Mlflow
        def sanitize_name(name):
            return re.sub(r"[^\w\-/ .]", "_", name)

        #Adding Data Drift Metrics into Mlflow
        report = Report([DataDriftPreset()])
        df_hist_new_eval = report.run(reference_df,df) 
        report_data = json.loads(df_hist_new_eval.json())

        mlflow.set_experiment("evidently_hist_vs_new")
        with mlflow.start_run():
            for i in report_data.get("metrics", []):
                metric_id = i.get("metric_id", "")
                value = i.get("value", None)

                if metric_id.startswith("Drifted"):
                    mlflow.log_metric("Number_of_driftedcolumns", value['count'])
                else:
                
                    clean_metric_id = sanitize_name(metric_id.lower().replace(" ", "_"))
                    mlflow.log_metric(clean_metric_id, value)
        return render_template('index.html', table=table_html)
    
    #returning the exception
    except Exception as e:
        return f"❌ An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
