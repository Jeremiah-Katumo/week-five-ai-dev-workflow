import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import shap

# Streamlit Page Setup
st.set_page_config(page_title="🩺 Patient Readmission Risk", layout="wide")
st.title("🏥 Patient Readmission Prediction (30 Days)")

uploaded_file = st.file_uploader("📤 Upload your cleaned patient CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Feature Engineering
    df['date_of_admission'] = pd.to_datetime(df['date_of_admission'])
    df['discharge_date'] = pd.to_datetime(df['discharge_date'])
    df = df.sort_values(['name', 'date_of_admission'])

    df['next_admission'] = df.groupby('name')['date_of_admission'].shift(-1)
    df['days_until_next_admission'] = (df['next_admission'] - df['discharge_date']).dt.days.abs()
    df['readmitted_30d'] = ((df['days_until_next_admission'] >= 0) &
                            (df['days_until_next_admission'] <= 30)).astype(int)
    df['admission_month'] = df['date_of_admission'].dt.month
    df['admission_dayofweek'] = df['date_of_admission'].dt.dayofweek
    df['length_of_stay'] = (df['discharge_date'] - df['date_of_admission']).dt.days

    drop_cols = ['name', 'doctor', 'hospital', 'date_of_admission', 'discharge_date',
                 'next_admission', 'days_until_next_admission']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    st.subheader("📋 Sample of Processed Data")
    st.dataframe(df.head())

    # Prepare data
    X = df.drop(columns='readmitted_30d')
    y = df['readmitted_30d']

    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessing
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    # Sidebar - model settings
    st.sidebar.header("⚙️ Model Settings")
    model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])
    sampling_choice = st.sidebar.radio("Class Imbalance Handling", ["None", "Oversample", "Undersample"])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    if sampling_choice == "Oversample":
        X_train, y_train = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
    elif sampling_choice == "Undersample":
        X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Model selection
    if model_choice == "Logistic Regression":
        model = Pipeline([
            ('pre', preprocessor),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
    elif model_choice == "Random Forest":
        model = Pipeline([
            ('pre', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    elif model_choice == "XGBoost":
        model = Pipeline([
            ('pre', preprocessor),
            ('clf', XGBClassifier(
                n_estimators=100,
                max_depth=8,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            ))
        ])

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.metric("🎯 Accuracy", f"{acc:.2%}")
    st.metric("📌 F1 Score", f"{f1:.2%}")

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.subheader("📊 Classification Report")
    st.dataframe(report_df.round(2))

    # Export report to CSV
    csv_download = report_df.to_csv().encode('utf-8')
    st.download_button("📥 Download Report CSV", csv_download, file_name="classification_report.csv", mime="text/csv")

    # Confusion Matrix
    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Readmit", "Readmit"],
                yticklabels=["No Readmit", "Readmit"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # ROC Curve
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], linestyle='--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # Precision-Recall Curve
    st.subheader("📉 Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision, label="Precision-Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR Curve")
    ax_pr.legend()
    st.pyplot(fig_pr)

    # SHAP Explanation (only for tree models)
    if model_choice == "XGBoost":
        st.subheader("🧠 SHAP Feature Importance")
        # Extract trained model and feature names
        booster = model.named_steps['clf']
        pre = model.named_steps['pre']
        X_enc = pre.fit_transform(X_train)
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_enc)

        # SHAP summary plot
        st.write("SHAP summary plot (top features):")
        fig_shap, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_values, X_enc, plot_type="bar", show=False)
        st.pyplot(fig_shap)
