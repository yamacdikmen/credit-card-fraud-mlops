import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# Load the preprocessed training and test datasets
train = pd.read_csv("data/creditcard_train.csv")
test = pd.read_csv("data/creditcard_test.csv")

# Separate features and target variable
X_train = train.drop(columns=["Class"])
y_train = train["Class"]
X_test = test.drop(columns=["Class"])
y_test = test["Class"]

# Apply standard scaling to normalize feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to handle class imbalance by generating synthetic minority class samples
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Start a new MLflow experiment for tracking metrics and model artifacts
mlflow.set_experiment("Credit Card Fraud Detection with SMOTE")

with mlflow.start_run():
    # Train logistic regression model on the SMOTE-balanced dataset
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_res, y_train_res)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics and model to MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "logistic_model_smote")

    # Print results for immediate feedback
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

