import joblib
import pandas as pd
import numpy as np
import os

# Load the model
model_path = "heart_model.pkl"
if not os.path.exists(model_path):
    print("Model file not found!")
    exit(1)

model = joblib.load(model_path)

# Define columns
features_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak']

# Test Case 1: Likely Low Risk
# Young, female, typical angina (low risk), normal BP, normal chol, no stress
low_risk_case = [30, 0, 1, 110, 180, 0, 0, 170, 0, 0]

# Test Case 2: Likely High Risk
# Older, male, asymptomatic chest pain (high risk), high BP, high chol, high stress, exercise angina
high_risk_case = [65, 1, 4, 160, 300, 1, 2, 110, 1, 3.5]

def predict(vals):
    df = pd.DataFrame([vals], columns=features_cols)
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    return pred, prob

pred_low, prob_low = predict(low_risk_case)
pred_high, prob_high = predict(high_risk_case)

print(f"Low Risk Case: Pred={pred_low}, Proba={prob_low}")
print(f"High Risk Case: Pred={pred_high}, Proba={prob_high}")

# Check dataset distribution
try:
    dataset_path = "dataset/heart.csv"
    if os.path.exists(dataset_path):
        df_data = pd.read_csv(dataset_path, names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'], na_values='?')
        df_data.dropna(inplace=True)
        df_data['target'] = df_data['target'].apply(lambda x: 1 if x > 0 else 0)
        print("\nDataset Target Distribution:")
        print(df_data['target'].value_counts(normalize=True))
    else:
        print("Dataset not found at dataset/heart.csv")
except Exception as e:
    print(f"Error reading dataset: {e}")
