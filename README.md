# 🧠 Credit Card Fraud Detection (ML + FastAPI)

An end-to-end **Machine Learning project** to detect fraudulent credit card transactions using classical ML models, optimized for **high fraud recall**, and deployed as a **FastAPI REST API**.

---

## 🚀 Project Overview

Credit card fraud detection is a highly **imbalanced classification problem**, where missing a fraud transaction is far more costly than flagging a normal one.

In this project, I:
- Trained multiple ML models
- Focused on **fraud recall** instead of accuracy
- Tuned the probability threshold for business impact
- Deployed the final model using **FastAPI**

---

## 📊 Dataset

- **Source:** Kaggle – Credit Card Fraud Detection  
- **Size:** 284,807 transactions  
- **Fraud cases:** 492 (≈ 0.17%)  
- **Features:**  
  - `V1` – `V28` (PCA transformed)  
  - `Time`, `Amount`  
  - Target: `Class` (0 = Normal, 1 = Fraud)

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Imbalanced-learn (SMOTE)**
- **XGBoost**
- **Matplotlib, Seaborn**
- **FastAPI**
- **Joblib**

---

## 🔍 Methodology

### 1️⃣ Exploratory Data Analysis (EDA)
- Analyzed class imbalance
- Studied transaction amount distribution

### 2️⃣ Data Preprocessing
- Feature scaling using `StandardScaler`
- Train-test split with stratification
- Class imbalance handled using **SMOTE**

### 3️⃣ Model Training
Trained and evaluated:
- Logistic Regression
- Random Forest
- XGBoost

### 4️⃣ Model Selection
- Compared models using:
  - **Fraud Recall**
  - **ROC-AUC**
- Selected the best model **programmatically** based on fraud recall

### 5️⃣ Threshold Tuning
- Default threshold (0.5) replaced with a tuned threshold
- Optimized to **maximize fraud recall**

---

## 📈 Model Performance

Due to extreme class imbalance, **accuracy alone is misleading**.  
Evaluation focused on recall and ROC-AUC.

### ✅ Training Performance
- ROC-AUC: **1.00**
- Fraud Recall: **1.00**

### ✅ Test Performance
- ROC-AUC: **~0.98**
- Fraud Recall: **~0.87**
- Accuracy: **~0.99** (expected due to imbalance)

> The model prioritizes minimizing **false negatives**, which is critical in real-world fraud detection systems.

---

## 🌐 FastAPI Deployment

The trained model is deployed as a REST API using **FastAPI**.

### 📦 Saved Artifacts
- `fraud_model.pkl` – trained ML model  
- `scaler.pkl` – preprocessing scaler  
- `threshold.pkl` – tuned decision threshold  

### ▶️ Run the API
```bash
uvicorn app:app --reload
```

### 📍 API Endpoints

#### Health Check
```
GET /
```

#### Predict Fraud
```
POST /predict
```

**Request Body (JSON):**
```json
{
  "features": [V1, V2, ..., V28, Time, Amount]
}
```

**Response:**
```json
{
  "fraud_probability": 0.87,
  "fraud_prediction": 1
}
```

---

## 🧠 Key Learnings

- Accuracy is unreliable for imbalanced datasets
- Fraud detection requires **recall-focused optimization**
- Threshold tuning is critical for business impact
- Deployment requires saving **model + preprocessing + decision logic**
- FastAPI enables clean and lightweight ML deployment

---

## 📌 Future Improvements

- Cost-sensitive learning
- Real-time streaming predictions
- Dockerized deployment
- Cloud hosting (AWS / Render)

---

## 👤 Author

**Ashutosh**  
Aspiring Data Scientist / ML Engineer  
Actively building real-world ML & NLP projects
