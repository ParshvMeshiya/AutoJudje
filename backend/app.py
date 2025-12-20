import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # backend/
ROOT_DIR = os.path.dirname(BASE_DIR)                    # project root
MODEL_DIR = os.path.join(ROOT_DIR, "models")            # models/
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer_v2.pkl"))
classifier = joblib.load(os.path.join(MODEL_DIR, "classifier_logreg.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
app = FastAPI(
    title="AutoJudge – Programming Problem Difficulty Predictor",
    description="Predicts difficulty class (easy / medium / hard) from problem text",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class ProblemInput(BaseModel):
    text: str
@app.get("/")
def health_check():
    return {"status": "AutoJudge backend running"}
@app.post("/predict")
def predict_difficulty(problem: ProblemInput):
    """
    Input: programming problem text
    Output: predicted difficulty class
    """
    X = tfidf.transform([problem.text])
    class_id = classifier.predict(X)[0]
    class_label = label_encoder.inverse_transform([class_id])[0]

    return {
        "difficulty_class": class_label
    }
