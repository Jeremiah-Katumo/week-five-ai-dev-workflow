# Patient Readmission Prediction Project

This repository contains code, notebooks, and applications for predicting the risk of patient readmission within 30 days post-discharge using healthcare data.

## Project Structure

```
week-five-ai-dev-workflow/
├── data/                # Raw and processed datasets
├── notebooks/           # Jupyter notebooks for EDA, preprocessing, modeling
├── apps/
│   ├── fastapi/         # FastAPI backend and React frontend
│   ├── flask/           # Flask backend and React frontend
│   └── streamlit/       # Streamlit app
├── requirements.txt     # Python dependencies
└── README.md            # Project overview
```

## Data Link

- Students Dropout and Academic Success - [View](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- HealthCare Dataset - [View](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)

## Overview

The goal of this project is to build and deploy machine learning models that help hospitals identify patients at high risk of readmission, enabling early intervention and better resource allocation.

## Main Components

- **Data Exploration & Preprocessing:**  
  Notebooks for cleaning, exploring, and engineering features from healthcare datasets.
- **Modeling:**  
  Training and evaluation of models such as Logistic Regression, Random Forest, and XGBoost.
- **Imbalance Handling:**  
  Techniques like oversampling, undersampling, and class weighting to address class imbalance.
- **APIs & Frontends:**  
  Deployable apps using FastAPI or Flask for serving predictions, with React-based frontends for user interaction.

## How to Use

1. **Set up the environment:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Explore and train models:**
    - Open and run the notebooks in the `notebooks/` directory.

3. **Run an API backend:**
    - See `apps/fastapi/README.md` or `apps/flask/README.md` for backend and frontend instructions.

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, imbalanced-learn, xgboost, janitor, fastapi, flask, react

## License

MIT License

---