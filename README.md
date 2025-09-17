# 🛡️ Explainable AI Fraud Detection Dashboard  
**Advanced Fraud Detection with Explainability using LightGBM, SHAP, and Streamlit**  

## 🎬 Demo

Visit the live demo at : https://explainable-ai-dashboard-for-fraud-detection.streamlit.app

---

## 📌 Overview  
This project implements an **Explainable AI (XAI) fraud detection system** using a **LightGBM classifier** on the IEEE Credit Card Fraud Detection dataset.  
It provides an interactive dashboard where users can analyze individual transactions, understand model decisions with **SHAP explanations** and **counterfactual suggestions**, and perform batch analysis for multiple transactions.

👉 Dataset:  
[IEEE-CIS Fraud Detection - Kaggle Dataset](https://www.kaggle.com/c/ieee-fraud-detection/data)

During preprocessing, extensive **feature engineering** was performed, including:  
- Extracting **TransactionHour**, **TransactionDate**, and **TransactionDayOfWeek** from the original `TransactionDT` column  
- Removing high-null or irrelevant columns like `dist2`, `D7`, `TransactionID`  
- Encoding categorical variables  
- Handling missing values  

---

## 🚀 Features  
✅ **Interactive Transaction Analysis**  
- Predict fraud vs legitimate transactions  
- Visual risk gauge  
- SHAP feature importance visualization  
- Counterfactual explanations suggesting actionable changes  

✅ **Model Performance Monitoring**  
- Confusion matrix  
- ROC & Precision-Recall curves  
- Classification report  
- Cost-Benefit Analysis  

✅ **Batch Analysis**  
- Analyze multiple transactions in one go  
- Probability distribution plots  
- False positive and false negative identification  

✅ **Explainable AI (XAI)**  
- SHAP TreeExplainer for feature-level explanations  
- Transparent decision-making for regulatory compliance and business trust  

---

## 🛠️ Installation  

1. **Clone the repository**  
```bash
git clone https://github.com/Arsalan80425/Explainable-AI-Dashboard-for-Fraud-Detection.git
cd Explainable-AI-Dashboard-for-Fraud-Detection
```

2. **Install dependencies**  
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage  

1. Ensure your dataset file is named `train_data.csv` and placed in the same directory as the code.  
   Download the dataset from [IEEE Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection/data).

2. Run the dashboard:  
```bash
streamlit run fraud_detection_dashboard.py
```

3. Interact via the web UI:  
- Load full dataset or sample  
- Perform individual or batch transaction analysis  
- Visualize SHAP explanations and counterfactual suggestions  
- View model performance metrics and cost-benefit analysis  

---

## 📂 Project Structure  
```
.
├── fraud_detection_dashboard.py      # Streamlit dashboard app
├── fraud_detection_model.pkl         # Lightgbm model
├── label_encoders.pkl                # Labels catogerical features into numerical features
├── shap_explainer.pkl                # explains features contribution to the prediction
├── feature_names.pkl                 # Track of Features
├── model_metrics.pkl                 # Model Report
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 📚 Tech Stack  
- **Machine Learning Model**: LightGBM  
- **Explainability**: SHAP (TreeExplainer)  
- **Data Handling**: Pandas, NumPy  
- **Web Interface**: Streamlit  
- **Visualization**: Plotly  

---

## ✅ Feature Engineering Highlights  
- Extracted time-based features from `TransactionDT`:  
   - TransactionHour (0–23)  
   - TransactionDate (date part)  
   - TransactionDayOfWeek (0=Monday, 6=Sunday)  
- Removed irrelevant features (high missingness or low predictive power)  
- Encoded categorical features using LabelEncoder  
- Filled missing numerical values with median  

---

## 🎯 Business Impact  
- Provides actionable counterfactual explanations  
- Transparent decisions to support audit and compliance  
- Reduces false positives to improve customer experience  
- Maximizes fraud detection rate to reduce financial loss  

---

## 📞 Contact  
- **Developer**: Mohammed Arsalan  
- **Email**: arsalanshaikh0408@gmail.com  
- **LinkedIn**: [LinkedIn Profile](http://www.linkedin.com/in/mohammed-arsalan-58543a305)  

---

## 👨‍💻 Author  
Mohammed Arsalan  
🎯 Passionate about combining AI, explainability, and real-world business applications for responsible and transparent decision-making.
