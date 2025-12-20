import sys
import os
import os
from scipy.sparse import hstack
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_engineering import (
    text_length,
    word_count,
    math_symbol_count,
    keyword_features,
    constraint_features,
    input_structure_features,
    algorithmic_depth
)
app = FastAPI(
    title="AutoJudge - Programming Problem Difficulty Predictor"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer_v2.pkl"))
clf = joblib.load(os.path.join(MODEL_DIR, "classifier_logreg.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
regressor = joblib.load(os.path.join(MODEL_DIR, "regressor_rf.pkl"))
class PredictRequest(BaseModel):
    text: str
@app.get("/")
def health():
    return {"status": "ok"}
@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        return {"error": "Empty input"}

    X_tfidf = tfidf.transform([text])

    manual = [
        text_length(text),
        word_count(text),
        math_symbol_count(text),
        algorithmic_depth(text),
        *keyword_features(text),
        *constraint_features(text),
        *input_structure_features(text)
    ]

    manual_np = np.array(manual).reshape(1, -1)
    X_final = hstack([X_tfidf, manual_np])

    # Classification
    probs = clf.predict_proba(X_final)[0]
    pred_class = clf.predict(X_final)[0]
    difficulty = label_encoder.inverse_transform([pred_class])[0]

    confidence = probs[pred_class]

    # Calibrated score
    ranges = {
        "easy": (1.0, 4.0),
        "medium": (4.0, 7.0),
        "hard": (7.0, 10.0)
    }

    lo, hi = ranges[difficulty]
    base_score = lo + confidence * (hi - lo)

    # Optional regression refinement
    try:
        reg_score = regressor.predict(X_final)[0]
        score = 0.8 * base_score + 0.2 * reg_score
    except:
        score = base_score

    return {
        "difficulty": difficulty,
        "confidence": round(float(confidence), 2),
        "problem_score": round(float(score), 2)
    }
