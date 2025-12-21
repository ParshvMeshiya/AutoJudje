# AutoJudge – Programming Problem Difficulty Predictor
AutoJudge is a **full-stack Machine Learning application** that predicts:
- **Difficulty class**: Easy / Medium / Hard  
- **Numerical difficulty score**: 0–10 

for programming problems using their **textual problem statements**.

The system combines NLP, feature engineering, classical ML models, and a FastAPI + React deployment pipeline.

## Features
- Accepts raw programming problem statements as input
- Predicts difficulty **class** (Easy / Medium / Hard)
- Predicts a **continuous difficulty score (0–10)**
- Uses **TF-IDF + handcrafted algorithmic features**
- Real-time inference via **FastAPI backend**
- Clean and interactive **React frontend**
- Fully **CORS-enabled** frontend ↔ backend communication
---

## Feature Engineering

Both **textual** and **structural** features are extracted:

### Text Features
- TF-IDF vectorization (5000 features)
- N-grams (1–2)
### Hand-crafted Features (27 total)
- Text length, word count, math symbol count
- Algorithmic keywords (DP, graph, tree, greedy, etc.)
- Constraint indicators (1e5, time limits, test cases)
- Input structure hints (matrix, graph, multiple test cases)

**Total feature size: 5027**
---

## Machine Learning Models

| Task | Model |
|-----|------|
| Difficulty Classification | Logistic Regression |
| Difficulty Score Regression | Random Forest Regressor |

---

## 📂 Project Structure

```text
AutoJudge/
│
├── backend/
│   └── app.py                  # FastAPI inference API
│
├── frontend/
│   ├── src/
│   │   └── App.js               # React UI
│   └── package.json
│
├── models/
│   ├── classifier_logreg.pkl    # Difficulty classifier
│   ├── regressor_rf.pkl         # Difficulty score regressor
│   ├── tfidf_vectorizer_v2.pkl  # TF-IDF vectorizer
│   └── label_encoder.pkl
│
├── utils/
│   └── feature_engineering.py   # Feature extraction logic
│
├── notebooks/
│   ├── 03_feature_extraction.ipynb
│   ├── 04_classification.ipynb
│   └── 05_regression.ipynb
│
├── data/
│   └── problems.csv             # Dataset
│
├── requirements.txt

└── README.md
```
---

## Setup and Installation

### (1)Clone the repository
```bash
git clone https://github.com/your-username/AutoJudge.git
cd AutoJudge
```

### (2️)Create and activate virtual environment
```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### (3️)Install backend dependencies
```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### (4)Running the Backend (FastAPI)
```bash
cd backend
uvicorn app:app --reload
```

Backend will be available at:

[http://127.0.0.1:8000](http://127.0.0.1:8000)


### (5)Running the Frontend (React)
```bash
cd frontend
npm install
npm start
```

Frontend will run at:

[http://localhost:3000](http://localhost:3000)

---
