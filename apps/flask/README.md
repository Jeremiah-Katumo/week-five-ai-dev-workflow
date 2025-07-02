# Flask App

## Folder Structure

patient-readmission-app/
├── backend/
│   ├── app.py
│   ├── utils.py
│   └── model/
│       └── predictor.pkl
└── frontend/
    ├── public/
    └── src/
        ├── App.jsx
        ├── index.js
        └── components/
            ├── UploadForm.jsx
            └── ResultsCard.jsx

---

## Overview

This project is a full-stack application for predicting patient readmission risk within 30 days post-discharge.  
It uses a Flask backend for model inference and a React frontend for user interaction.

## Features

- Upload patient data and receive readmission risk predictions.
- REST API built with Flask.
- Frontend built with React.
- Pre-trained ML model for inference.

## Getting Started

### Backend

1. **Install dependencies:**
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

2. **Run the Flask server:**
    ```bash
    python app.py
    ```

3. **Model file:**  
   Ensure `predictor.pkl` is present in the `model/` directory.

### Frontend

1. **Install dependencies:**
    ```bash
    cd frontend
    npm install
    ```

2. **Run the React app:**
    ```bash
    npm start
    ```

3. The app will be available at [http://localhost:3000](http://localhost:3000).

## API Endpoints

- `POST /predict`  
  Accepts patient data and returns readmission risk prediction.

## File Descriptions

- **backend/app.py**: Flask entry point.
- **backend/utils.py**: Utility functions for preprocessing and prediction.
- **backend/model/predictor.pkl**: Pre-trained ML model.
- **frontend/src/**: React source code.
- **frontend/src/components/**: UI components.

## Notes

- Ensure the backend is running before using the frontend.
- Update the API URL in the frontend if backend runs on a different host/port.

## License

MIT License

---