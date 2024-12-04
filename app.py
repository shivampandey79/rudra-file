# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Display dataset
    st.title("Wine Quality Analysis App")
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Data Exploration
    st.subheader("Dataset Information")
    st.write(df.info())
    st.write(df.describe().T)

    # Missing Values
    st.subheader("Handling Missing Values")
    st.write("Missing Values per Column:")
    st.write(df.isnull().sum())

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    st.write("Total Missing Values after Imputation:", df.isnull().sum().sum())

    # Histogram
    st.subheader("Histogram")
    st.pyplot(df.hist(bins=20, figsize=(10, 10)))

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 12))
    sb.heatmap(corr > 0.7, annot=True, cbar=False, ax=ax)
    st.pyplot(fig)

    # Feature Engineering
    st.subheader("Feature Engineering")
    if 'total sulfur dioxide' in df.columns:
        df = df.drop('total sulfur dioxide', axis=1)

    df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
    features = df.drop(['quality', 'best quality'], axis=1)
    target = df['best quality']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

    # Train-Test Split
    xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)

    # Normalize Features
    norm = MinMaxScaler()
    xtrain = norm.fit_transform(xtrain)
    xtest = norm.transform(xtest)

    # Model Training and Evaluation
    st.subheader("Model Training and Evaluation")
    models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
    model_names = ["Logistic Regression", "XGBoost Classifier", "SVM"]

    for i, model in enumerate(models):
        model.fit(xtrain, ytrain)
        train_accuracy = metrics.roc_auc_score(ytrain, model.predict(xtrain))
        val_accuracy = metrics.roc_auc_score(ytest, model.predict(xtest))
        st.write(f"**{model_names[i]}**")
        st.write(f"Training Accuracy: {train_accuracy}")
        st.write(f"Validation Accuracy: {val_accuracy}")

    # Confusion Matrix
    st.subheader("Confusion Matrix for XGBoost")
    cm = confusion_matrix(ytest, models[1].predict(xtest))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=models[1].classes_)
    disp.plot(ax=ax)
    st.pyplot(fig)
else:
    st.write("Please upload a CSV file to start.")
