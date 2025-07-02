# FastAPI App

## Project Structure

patient-readmission-app/
├── backend/
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   ├── utils.py
│   ├── model/
│   │   └── predictor.pkl
│   └── requirements.txt
├── frontend/
│   ├── public/
│   └── src/
│       ├── App.jsx
│       ├── components/
│       │   ├── UploadForm.jsx
│       │   └── ResultsCard.jsx
│       ├── services/api.js
│       └── index.js
├── README.md

---

## Overview

This project is a full-stack application for predicting patient readmission risk within 30 days post-discharge.  
It uses a FastAPI backend for model inference and a React frontend for user interaction.

## Features

- Upload patient data and receive readmission risk predictions.
- REST API built with FastAPI.
- Frontend built with React.
- Pre-trained ML model for inference.

## Getting Started

### Backend

1. **Install dependencies:**
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

2. **Run the FastAPI server:**
    ```bash
    uvicorn main:app --reload
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

- **backend/main.py**: FastAPI entry point.
- **backend/models.py, schemas.py**: Data models and validation.
- **backend/utils.py**: Utility functions for preprocessing and prediction.
- **backend/model/predictor.pkl**: Pre-trained ML model.
- **frontend/src/**: React source code.
- **frontend/src/components/**: UI components.
- **frontend/src/services/api.js**: API calls to backend.

## Notes

- Ensure the backend is running before using the frontend.
- Update the API URL in `frontend/src/services/api.js` if backend runs on a different host/port.

## License

MIT License