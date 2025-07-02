#!/usr/bin/env python
# coding: utf-8

# ## Problem: Predict the risk of patient readmission within 30 days post-discharge.
# 
# ### Objectives:
# - Reduce avoidable readmissions.
# - Support physicians with early intervention recommendations.
# - Optimize hospital resource allocation.
# 
# ### Stakeholders:
# - Doctors and medical staff
# - Hospital administrators
# 
# ## Data Strategy
# ### Data Sources:
# - Electronic Health Records (EHRs): lab results, vitals, medications
# - Patient demographics and prior admission history
# 
# ### Two Ethical Concerns:
# - **Patient Privacy**: Sensitive health data must be protected (HIPAA-compliant handling).
# - **Bias**: Historical disparities (e.g., based on insurance or race) may affect prediction fairness.
# 
# ### Preprocessing Pipeline:
# - **1. Data Cleaning**: Handle missing vitals/lab data (e.g., impute using median).
# - **2. Feature Engineering**:
#     - Calculate time since last admission
#     - Number of comorbidities
# - **3. Encoding**:
#     - One-hot encode categorical features
#     - Scale numeric features using MinMaxScaler
# 

# In[98]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import janitor

# %matplotlib_inline
sns.set_theme(style='whitegrid')


# In[99]:


patients = pd.read_csv('../data/healthcare_dataset.csv', sep=",")
patients = janitor.clean_names(patients)
patients


# In[100]:


patients.columns.to_list()


# In[101]:


patients['duplicate_count'] = patients.groupby(['name'])['name'].transform('count')
patients


# In[102]:


patients.columns.tolist()


# In[103]:


# adrIENNE bEll
patients[patients['name'] == 'adrIENNE bEll']


# In[104]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[105]:


# Ensure dates are datetime
patients['date_of_admission'] = pd.to_datetime(patients['date_of_admission'])
patients['discharge_date'] = pd.to_datetime(patients['discharge_date'])
patients.head()


# In[106]:


# Sort by patient and admission date
patients = patients.sort_values(['name', 'date_of_admission'])

# Create a column for the next admission date for each patient
patients['next_admission'] = patients.groupby('name')['date_of_admission'].shift(-1)
patients


# In[107]:


# Calculate days until next admission
patients['days_until_next_admission'] = (patients['next_admission'] - patients['discharge_date']).dt.days
patients


# In[108]:


patients['days_until_next_admission'] = patients['days_until_next_admission'].abs()
patients


# In[109]:


# Binary target: 1 if next admission within 30 days, else 0
patients['readmitted_30d'] = ((patients['days_until_next_admission'] >= 0) & 
                              (patients['days_until_next_admission'] <= 30)).astype(int)
patients


# In[110]:


patients['admission_month'] = patients['date_of_admission'].dt.month
patients['admission_dayofweek'] = patients['date_of_admission'].dt.dayofweek


# In[111]:


patients.columns.to_list()


# In[112]:


patients['length_of_stay'] = (patients['discharge_date'] - patients['date_of_admission']).dt.days


# In[113]:


patients[['name', 'date_of_admission', 'discharge_date', 'readmitted_30d', 'admission_month',
 'admission_dayofweek', 'length_of_stay']]


# In[114]:


# Drop helper columns if you want
helper_columns = [
   'name', 'doctor', 'hospital', 'date_of_admission', 'discharge_date', 
   'duplicate_count', 'next_admission', 'days_until_next_admission'
]
patients[helper_columns].head()


# In[115]:


# Drop helper columns
patients = patients.drop(columns=helper_columns)
patients


# In[116]:


patients.columns.to_list()


# In[117]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


# In[118]:


# Optional: XGBoost (install if needed)
try:
    from xgboost import XGBClassifier
    xgb_installed = True
except ImportError:
    xgb_installed = False

# 1. Select columns
# selected_columns = [
#     'name', 'age', 'gender', 'blood_type', 'medical_condition', 'date_of_admission',
#     'doctor', 'hospital', 'insurance_provider', 'billing_amount', 'room_number',
#     'admission_type', 'discharge_date', 'medication', 'test_results', 'duplicate_count'
# ]
# patients_selected = patients[selected_columns].copy()


# In[119]:


patients.info()


# In[133]:


X = patients.drop(columns=['readmitted_30d'])
y = patients['readmitted_30d']

# 5. Identify numeric and categorical columns
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# 6. Preprocessing pipeline
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


# In[134]:


# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[135]:


X_train.columns


# In[136]:


y_train = pd.DataFrame(y_train)
y_train.columns


# In[137]:


# 8. Fit and evaluate models

# Logistic Regression
log_reg = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])

log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_log_reg))


# In[138]:


# Random Forest
rf_clf = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42))
])

rf_clf.fit(X_train, y_train)
y_pred_rf_clf = rf_clf.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf_clf))


# In[141]:


# XGBoost (if installed)
if xgb_installed:
    xgb = Pipeline([
        ('pre', preprocessor),
        ('clf', XGBClassifier(n_estimators=100, max_depth=10, use_label_encoder=True, eval_metric='logloss', random_state=42))
    ])
    
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    print("XGBoost Results:")
    print(classification_report(y_test, y_pred_xgb))
else:
    print("XGBoost not installed. Skipping XGBoost model.") 


# In[142]:


if xgb_installed:
    xgb = Pipeline([
        ('pre', preprocessor),
        ('clf', XGBClassifier(n_estimators=100, max_depth=10, use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    print("XGBoost Results:")
    print(classification_report(y_test, y_pred_xgb))
else:
    print("XGBoost not installed. Skipping XGBoost model.")


# ## What do these numbers mean?
# - **Class 0 (not readmitted within 30 days):**
#     - Precision: 0.90 (90% of predicted 0s are correct)
#     - Recall: 0.96 (96% of actual 0s are found)
#     - F1-score: 0.93 (harmonic mean of precision and recall)
#     - Support: 10,000 samples
#     - Class 1 (readmitted within 30 days):
# 
# - **Precision: 0.00**
#     - Recall: 0.00
#     - F1-score: 0.00
#     - Support: 1,100 samples
# - **Overall accuracy:** 0.87 (87% of all predictions are correct)
# - **Macro avg:** Average of metrics for both classes, treating them equally.
# - **Weighted avg:** Average of metrics weighted by the number of samples in each class.
# 
# ## Interpretation
# - **The model predicts almost all samples as class 0.**
# - **It fails to identify any class 1 cases** (readmissions within 30 days): precision, recall, and F1-score are all 0 for class 1.
# - **High accuracy (0.87) is misleading** because the dataset is imbalanced (much more class 0 than class 1).
# - **Macro and weighted averages are low** due to the model's inability to predict class 1.
# 
# ## What does this mean?
# - **The model is not learning to detect readmissions (class 1) at all.**
# - This is a classic case of **class imbalance**: the model is biased toward the majority class (class 0).
# - **Action:**
#     - Try resampling techniques (oversample class 1, undersample class 0)
#     - Use class weights in your model
#     - Try different algorithms or hyperparameters

# In[170]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Oversample the minority class (class 1)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)


# In[171]:


# Use Class Weights
# for scikit-learn models
log_reg = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# for XGBoost 
# Convert y_train to a NumPy array or Series before the calculation:
# If y_train is a DataFrame, convert to Series
if isinstance(y_train, pd.DataFrame):
    y_train_series = y_train.squeeze()
else:
    y_train_series = y_train
# Calculate scale_pos_weight
# scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = Pipeline([
    ('pre', preprocessor),
    ('clf', XGBClassifier(n_estimators=100, max_depth=10, eval_metric='logloss', random_state=42, 
                          scale_pos_weight=np.array(scale_pos_weight)[0]))
])


# In[172]:


print(scale_pos_weight.dtype)
print(np.array(scale_pos_weight)) # np.array(scale_pos_weight)[0]


# In[173]:


# Logistic Regression with resampled data
log_reg.fit(X_resampled, y_resampled)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression Results (resampled):")
print(classification_report(y_test, y_pred_log_reg))


# In[174]:


# XGBoost with resampled data
xgb.fit(X_resampled, y_resampled)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Results (resampled):")
print(classification_report(y_test, y_pred_xgb))


# In[175]:


# Undersample the majority class (class 0)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)


# In[176]:


# Logistic Regression with resampled data
log_reg.fit(X_resampled, y_resampled)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression Results (resampled):")
print(classification_report(y_test, y_pred_log_reg))


# In[177]:


# XGBoost with resampled data
xgb.fit(X_resampled, y_resampled)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Results (resampled):")
print(classification_report(y_test, y_pred_xgb))


# - Models trained after oversampling perform a little much better as compared to models trained after undersampling the denominating class.

# In[ ]:




