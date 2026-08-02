# 🏠 House Price Predictor

An end-to-end **Machine Learning web application** that predicts residential house prices in **Hyderabad, Telangana, India**, based on property attributes such as location, area, BHK, floor number, and property age.

Built as a complete, production-quality AI/ML portfolio project — from data generation and preprocessing, through model training and evaluation, to a polished interactive Streamlit dashboard.

---

## 📌 Project Overview

House Price Predictor uses a supervised regression model trained on Hyderabad residential property data to estimate market prices. It compares multiple ML algorithms, automatically selects the best-performing one, and serves predictions through a modern, responsive Streamlit UI.

The app also compares the predicted price against the selected locality's average price and offers a simple investment classification (**Good Investment / Average Investment / Premium Property**).


## 🔗 Live Demo

🚀 **Try it now:** [hyderabad-house-price-ai.streamlit.app](https://hyderabad-house-price-ai.streamlit.app/)

Enter property details like location, area, BHK, and floor to get an instant AI-powered price estimate for residential properties across 20 major Hyderabad localities.

---

## ✨ Features

- 🧹 Full data cleaning pipeline — duplicate removal, missing value handling, outlier detection (IQR method)
- 🛠️ Feature engineering — floor ratio, label encoding, one-hot encoding of localities
- 🤖 Trains and compares **4–5 regression models**: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, (optionally XGBoost)
- 🏆 Automatic best-model selection based on **R² Score**
- 📊 Model evaluation with **MAE, MSE, RMSE, R²**
- 📈 Rich Matplotlib/Seaborn visualizations (distribution, correlation heatmap, feature importance, locality pricing, etc.)
- 💻 Modern Streamlit dashboard with custom CSS, cards, metrics, and a dark sidebar
- 💰 Indian Rupee formatted output (e.g. `₹ 82,50,000`)
- 📍 Locality price comparison and investment suggestion
- 🔄 Reset button and full input validation / exception handling

---

## 🧰 Technology Used

| Layer | Technology |
|---|---|
| Language | Python 3.14.6 |
| ML / Data | Scikit-learn, Pandas, NumPy, Joblib |
| Visualization | Matplotlib, Seaborn |
| Frontend | Streamlit |
| IDE | VS Code |

---

## 🐍 Python Version

This project targets **Python 3.14.6**. The codebase uses only standard, version-stable syntax, so it also runs correctly on Python 3.10+ if 3.14.6 is not yet available on your system.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/House-Price-Predictor.git
cd House-Price-Predictor

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1. Generate the dataset (already included, but can be regenerated)
```bash
python dataset/generate_dataset.py
```

### 2. Train the model
```bash
python train_model.py
```
This will:
- Clean and preprocess the dataset
- Train and compare all models
- Save the best model to `model.pkl`
- Save evaluation charts to `images/`
- Save the cleaned dataset to `dataset/cleaned_dataset.csv`

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```
Then open the local URL shown in your terminal (usually `http://localhost:8501`).

---

## 📁 Folder Structure

```
House-Price-Predictor/
│
├── app.py                          # Streamlit dashboard
├── train_model.py                  # ML training pipeline
├── utils.py                        # Shared helper functions
├── model.pkl                       # Saved best model (generated)
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   ├── generate_dataset.py         # Synthetic dataset generator
│   ├── hyderabad_house_prices.csv  # Raw dataset
│   └── cleaned_dataset.csv         # Cleaned dataset (generated)
│
├── notebooks/
│   └── EDA.ipynb                   # Exploratory Data Analysis notebook
│
├── images/                         # Generated charts (after training)
│
└── assets/                         # Screenshots / logos
```

---

## 📸 Screenshots
>### Dashboard
![Dashboard](assets/dashboard.png)
### Prediction Result
![Prediction Result](assets/prediction_result.png)

---

## 📊 Model Performance (on synthetic dataset)

| Model | R² Score | RMSE |
|---|---|---|
| **Gradient Boosting** ✅ | **0.9569** | ₹10,05,349 |
| Random Forest | 0.9463 | ₹11,21,844 |
| Linear Regression | 0.9224 | ₹13,48,167 |
| Decision Tree | 0.8962 | ₹15,59,595 |

*(Your exact numbers may vary slightly depending on random seeds, library versions, and whether you use a real dataset.)*

---

## 🚀 Future Improvements

- Integrate a real, verified Hyderabad real-estate dataset (e.g. from MagicBricks / 99acres, with proper licensing)
- Add hyperparameter tuning (GridSearchCV / Optuna)
- Add map-based locality selection
- Deploy to Streamlit Community Cloud / Render / HuggingFace Spaces
- Add authentication and price-history tracking
- Add SHAP-based model explainability

---

## 📄 License

This project is released under the **MIT License** — free to use, modify, and distribute with attribution.

---

## 👨‍💻 Author

**House Price Predictor Project**
Built as part of an AI/ML Internship portfolio.

---

⭐ If you find this project useful, consider giving it a star on GitHub!
