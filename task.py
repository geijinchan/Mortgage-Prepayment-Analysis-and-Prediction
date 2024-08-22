import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def download_and_load_data(url, extracted_dir, csv_file_path):
    if os.path.exists(csv_file_path):
        print(f"[INFO] The file '{csv_file_path}' already exists. No download needed.")
    else:
        print(f"[INFO] File not found. Downloading the ZIP file...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(extracted_dir)
            print(f"[INFO] The file has been downloaded and extracted to the '{extracted_dir}' directory.")
        else:
            print(f"[ERROR] Failed to download the file. Status code: {response.status_code}")

    
    data = pd.read_csv(csv_file_path)
    print('Data Loaded')
    return data

def preprocess_data(data):
    # Handling missing values
    data['SellerName'] = data['SellerName'].mode()[0]

    # Correcting data types
    data['FirstPaymentDate'] = pd.to_datetime(data['FirstPaymentDate'], format='%Y%m')
    data['MaturityDate'] = pd.to_datetime(data['MaturityDate'], format='%Y%m')

    # Handling categorical columns
    categorical_columns = ['FirstTimeHomebuyer', 'Occupancy', 'ProductType', 'PropertyState', 
                           'PropertyType', 'LoanPurpose', 'SellerName', 'ServicerName',
                           'NumBorrowers','PPM','Channel']
    
    le = LabelEncoder()
    for items in categorical_columns:
        data[items] = le.fit_transform(data[items])

    # Clean the 'MSA' column
    data['MSA'] = data['MSA'].str.strip()
    data['MSA'] = pd.to_numeric(data['MSA'], errors='coerce')
    data['MSA'].fillna(data['MSA'].mode().iloc[0], inplace=True)

    # Drop irrelevant columns
    data.drop(columns=['ProductType', 'SellerName', 'LoanSeqNum', 'PostalCode'], inplace=True)

    # Feature engineering
    data['LoanTermRemaining'] = data['MaturityDate'].dt.year - data['FirstPaymentDate'].dt.year

    print("Data Cleaning done")
    return data

def feature_selection(data):
    X = data.drop(columns=['EverDelinquent', 'FirstPaymentDate', 'MaturityDate'])
    y = data['EverDelinquent']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    rfe = RFE(model, n_features_to_select=10)
    fit = rfe.fit(X_scaled, y)

    feature_names = X.columns.tolist()  
    selected_features = [name for name, selected in zip(feature_names, fit.support_) if selected]
    
    print("Important Feature selection done")
    return selected_features, X[selected_features], y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print("-" * 60)

'''def perform_cross_validation(X, y):
    model = LogisticRegression(max_iter=1000)
    rf_model = RandomForestClassifier(random_state=42)

    # Perform cross-validation for Logistic Regression
    cv_scores_log = cross_val_score(model, X, y, cv=5)
    print("Logistic Regression Cross-Validation Scores:", cv_scores_log)
    print("Average CV Score:", cv_scores_log.mean())

    # Perform cross-validation for Random Forest
    cv_scores_rf = cross_val_score(rf_model, X, y, cv=5)
    print("Random Forest Cross-Validation Scores:", cv_scores_rf)
    print("Average CV Score:", cv_scores_rf.mean())

def plot_training_testing_split(X_train, X_test):
    plt.figure(figsize=(6, 4))
    plt.bar(['Training Set', 'Testing Set'], [len(X_train), len(X_test)], color=['blue', 'orange'])
    plt.ylabel('Number of Samples')
    plt.title('Training and Testing Set Split')
    plt.show()'''

if __name__ == "__main__":
    url = "https://github.com/Technocolabs100/Mortgage-Prepayment-Analysis-and-Prediction/archive/refs/heads/main.zip"
    extracted_dir = "dataset"
    csv_file_path = os.path.join(extracted_dir, "LoanExport.csv")

    data = download_and_load_data(url, extracted_dir, csv_file_path)
    data = preprocess_data(data)
    selected_features, X, y = feature_selection(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    train_and_evaluate_models(X_train, X_test, y_train, y_test)
    #perform_cross_validation(X, y)
    
    #plot_training_testing_split(X_train, X_test)
