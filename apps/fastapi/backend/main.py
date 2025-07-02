from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
from io import BytesIO
from utils import preprocess_data, predict_and_explain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs('model', exist_ok=True)
model = joblib.load("model/xgb.pkl")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

    processed_df = preprocess_data(df)
    results = predict_and_explain(model, processed_df)

    return results
