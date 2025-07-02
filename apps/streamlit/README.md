# Streamlit App

This directory contains a Streamlit application for predicting patient readmission risk within 30 days post-discharge.

## Folder Structure

```
streamlit/
├── app.py              # Main Streamlit app
├── model/
│   └── predictor.pkl   # Pre-trained ML model
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Overview

The Streamlit app provides an interactive web interface for uploading patient data and receiving readmission risk predictions using a pre-trained machine learning model.

## Features

- User-friendly web interface
- Upload patient data (CSV or manual entry)
- Instant prediction of readmission risk
- Visualization of input data and results

## Getting Started

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

3. **Model file:**  
   Ensure `predictor.pkl` is present in the `model/` directory.

4. **Access the app:**  
   Open your browser and go to [http://localhost:8501](http://localhost:8501).

## Notes

- The backend model must be trained and saved as `predictor.pkl` before running the app.
- Update the app as needed to match your data schema and features.

## License

MIT License

---