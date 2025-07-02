import pandas as pd
import shap

def preprocess_data(df):
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

    return df

def predict_and_explain(model, df):
    y_pred = model.predict(df)
    y_prob = model.predict_proba(df)[:, 1]
    explainer = shap.Explainer(model.named_steps['clf'])
    shap_values = explainer(model.named_steps['pre'].transform(df))
    top_features = pd.DataFrame(shap_values.values, columns=shap_values.feature_names).mean().abs().sort_values(ascending=False).head(5)

    return {
        "prediction_summary": {
            "positive": int(y_pred.sum()),
            "total": int(len(y_pred)),
            "probabilities": y_prob.tolist()
        },
        "important_features": top_features.to_dict()
    }