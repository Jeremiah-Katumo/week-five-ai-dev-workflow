# Notebooks

This directory contains Jupyter notebooks used for data exploration, preprocessing, feature engineering, model training, and evaluation for the patient readmission prediction project.

## Contents

- **Data Exploration:**  
  Initial analysis, visualization, and understanding of the healthcare dataset.

- **Preprocessing:**  
  Data cleaning, handling missing values, encoding categorical variables, and feature scaling.

- **Feature Engineering:**  
  Creation of new features such as time since last admission, length of stay, and duplicate counts.

- **Modeling:**  
  Training and evaluation of various machine learning models (Logistic Regression, Random Forest, XGBoost) for predicting 30-day patient readmission.

- **Imbalance Handling:**  
  Techniques such as oversampling, undersampling, and class weighting to address class imbalance in the target variable.

- **Interpretation:**  
  Analysis of model performance metrics and discussion of results.

## Usage

Open the notebooks in this directory with Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost, janitor

Install requirements with:

```bash
pip install -r ../requirements.txt
```

## Notes

- Notebooks are organized to follow the typical data science workflow.
- Outputs and results may vary depending on the dataset and random seeds.

## License

MIT License