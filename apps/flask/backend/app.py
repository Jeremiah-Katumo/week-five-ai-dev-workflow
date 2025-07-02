
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from utils import preprocess_data, predict_and_explain
from io import BytesIO

app = Flask(__name__)
CORS(app)

model = joblib.load("model/xgb.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    df = pd.read_csv(BytesIO(file.read()))
    df = preprocess_data(df)
    result = predict_and_explain(model, df)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)