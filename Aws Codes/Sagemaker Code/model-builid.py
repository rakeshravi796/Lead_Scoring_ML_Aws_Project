import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import sagemaker

# Your provided code
bucket = '**********'
input_s3_path = 's3://*********/train_input_test/Lead_Scoring_training.csv'
region = boto3.Session().region_name
role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.Session()
mlflow.set_tracking_uri("arn:aws:sagemaker:ap-south-1:********:mlflow-tracking-server/final-project-tracking-server")  # Replace with your MLflow Tracking URI
mlflow.set_experiment("LeadScoringModelComparison")
model_output = f's3://{bucket}/model-outputs/' 
df = pd.read_csv(input_s3_path)
X = df.drop(columns=['Converted'])
y = df['Converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate model
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Hyperparameter grids
xgb_param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300]
}

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Tune and Log XGBoost Model with Grid Search
with mlflow.start_run(run_name="XGBoost_GridSearch") as run:
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=3, scoring='f1', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)
    metrics_xgb = evaluate_model(y_test, y_pred_xgb)
    
    # Log best params, metrics, and model
    mlflow.log_params(xgb_grid.best_params_)
    mlflow.log_metrics(metrics_xgb)
    mlflow.xgboost.log_model(best_xgb, "xgboost_model")
    xgb_run_id = run.info.run_id
    print(f"XGBoost tuned with run_id: {xgb_run_id}, Best F1: {metrics_xgb['f1']}")

# Tune and Log RandomForest Model with Grid Search
with mlflow.start_run(run_name="RandomForest_GridSearch") as run:
    rf_model = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    metrics_rf = evaluate_model(y_test, y_pred_rf)
    
    # Log best params, metrics, and model
    mlflow.log_params(rf_grid.best_params_)
    mlflow.log_metrics(metrics_rf)
    mlflow.sklearn.log_model(best_rf, "randomforest_model")
    rf_run_id = run.info.run_id
    print(f"RandomForest tuned with run_id: {rf_run_id}, Best F1: {metrics_rf['f1']}")

# Compare Tuned Models and Register the Best One
# Load runs for comparison
xgb_run = mlflow.get_run(xgb_run_id)
rf_run = mlflow.get_run(rf_run_id)

# Compare based on F1-score
best_model = "xgboost" if xgb_run.data.metrics['f1'] > rf_run.data.metrics['f1'] else "randomforest"
best_run_id = xgb_run_id if best_model == "xgboost" else rf_run_id
best_model_uri = f"runs:/{best_run_id}/{best_model}_model"

print(f"Best tuned model: {best_model} with F1-score: {max(xgb_run.data.metrics['f1'], rf_run.data.metrics['f1'])}")

