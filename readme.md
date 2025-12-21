## Author

Parshv Meshiya  
B.Tech CSE, IIT Roorkee
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



## Prerequisites

- Python 3.8 or higher
- Node.js & npm
- Git

---
## Contribution

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

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

## Frontend Preview

1. Paste a programming problem statement
2. Click **Predict Difficulty**
3. Instantly view:
	- Difficulty label
	- Numerical difficulty score

---

## Model Performance (Approx.)

- **Classification Accuracy:** ~47%
- **Regression MAE:** ~1.8
- **Regression R²:** ~0.13


---

## Known Limitations

- Medium class is harder to separate due to dataset imbalance
- Regression scores may cluster around dataset mean

**Future improvements:**
- Better calibration of difficulty score
- Transformer-based embeddings
- Class-aware loss functions

---

## Tech Stack

- Python 3.9+
- FastAPI
- scikit-learn
- NumPy, SciPy
- React (Create React App)
