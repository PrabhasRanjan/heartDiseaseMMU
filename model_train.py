import pandas as pd
import numpy as np
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def download_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        print("Downloading UCI Heart Disease Dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        urllib.request.urlretrieve(url, dataset_path)
    
def prepare_data(dataset_path):
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(dataset_path, names=cols, na_values='?')
    df.dropna(inplace=True)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak']
    X = df[features]
    y = df['target']
    return X, y, features

def main():
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, "heart.csv")
    download_dataset(dataset_path)
    
    X, y, feature_names = prepare_data(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Better hyperparameters for Random Forest
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        min_samples_split=5, 
        random_state=42, 
        class_weight='balanced' # Help with any minor imbalance
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=feature_names)
    print("\nFeature Importances:")
    print(feat_importances.sort_values(ascending=False))
    
    joblib.dump(model, "heart_model.pkl")
    print("\nModel saved successfully!")

if __name__ == "__main__":
    main()
