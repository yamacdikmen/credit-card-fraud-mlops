# Credit Card Fraud Detection 🛡️

This project demonstrates a full ML Ops pipeline for detecting fraudulent credit card transactions using logistic regression and SMOTE.

## 📂 Project Structure
- `src/` → Training script with MLflow logging  
- `data/` → Contains `creditcard_test.csv` (training set excluded due to size)  
- `register_and_monitor.py` → Model registry + performance monitoring  
- `request.py` → REST API test script  
- `requirements.txt` → All necessary packages  

## 🚀 Technologies Used
- Python  
- Scikit-learn  
- MLflow  
- SMOTE (imblearn)  
- REST API (via MLflow model serving)  

## 📈 MLflow Metrics Logged
- Accuracy  
- Precision  
- Recall  
- F1 Score  

## 🧠 Model Explanation
- Logistic Regression is used due to its simplicity and interpretability.  
- SMOTE is applied to handle severe class imbalance.  
- Features `V1`–`V28` are PCA-transformed. `Amount` and `Time` are also used.  

## 🌐 API Deployment
Model is served locally using:
```bash
mlflow models serve -m "runs:/<RUN_ID>/logistic_model_smote" -p 5001
```

Send POST requests to:
```
http://127.0.0.1:5001/invocations
```

## ⚙️ How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python src/train.py

# 3. Monitor / log drift
python register_and_monitor.py

# 4. Serve the model (in new terminal)
mlflow models serve -m "runs:/<RUN_ID>/logistic_model_smote" -p 5001

# 5. Send prediction request
python request.py
```

## 📂 Dataset
Original dataset: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

`creditcard_train.csv` was excluded from Git due to size (>100MB).

## 👤 Author
Yamaç Dikmen  
Bahçeşehir University – AI Engineering  
April 2025
