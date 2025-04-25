import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load test data
df = pd.read_csv("data/creditcard_test.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

# Scale test data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load model from run
run_id = "d5a753ba346c428cb17eb969fc9641d8"
logged_model_uri = f"runs:/{run_id}/logistic_model_smote"
model = mlflow.sklearn.load_model(logged_model_uri)

# Predict and calculate metrics
y_pred = model.predict(X_scaled)
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Log drift check as new run
mlflow.set_experiment("Credit Card Fraud Drift Monitoring")
with mlflow.start_run(run_name="drift_check_" + run_id):
    mlflow.log_param("source_run_id", run_id)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

# Register the model
client = MlflowClient()
model_name = "CreditCardFraudModel"

# Try registering model only if not already present
try:
    client.create_registered_model(model_name)
except:
    pass  # already exists

# Create a model version and transition to Staging
model_uri = f"runs:/{run_id}/logistic_model_smote"
client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
client.transition_model_version_stage(name=model_name, version=1, stage="Staging")

print("✅ Model registered and transitioned to 'Staging'.")
print("✅ Drift metrics logged.")
