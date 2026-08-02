"""
train_model.py
---------------
End-to-end machine learning pipeline for the House Price Predictor project.

Pipeline steps:
    1. Load raw dataset
    2. Clean data (duplicates, missing values, outliers)
    3. Feature engineering & encoding
    4. Train / test split + scaling
    5. Train multiple regression models
    6. Evaluate each model (MAE, MSE, RMSE, R2)
    7. Select the best model automatically (highest R2)
    8. Save the best model + preprocessing objects with Joblib
    9. Generate evaluation & EDA visualizations (saved to images/)

Run:
    python train_model.py

Author: House Price Predictor Project
Python: 3.14.6
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from utils import clean_dataset, NUMERIC_COLUMNS, ensure_dir

DATA_PATH = "dataset/hyderabad_house_prices.csv"
MODEL_PATH = "model.pkl"
IMAGES_DIR = "images"

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (9, 6)


# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ------------------------------------------------------------------
# 2 & 3. Clean + feature engineer + encode
# ------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    print("\n--- Preprocessing ---")
    print(f"Missing values before cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Duplicate rows before cleaning: {df.duplicated().sum()}")

    df_clean = clean_dataset(df)
    print(f"Rows after cleaning: {df_clean.shape[0]} (removed {df.shape[0] - df_clean.shape[0]})")

    # Feature engineering: floor ratio (relative position in building)
    df_clean["Floor_Ratio"] = (
        df_clean["Floor_Number"] / df_clean["Total_Floors"].replace(0, 1)
    ).round(3)

    # Label-encode Location (also usable for tree models directly)
    le = LabelEncoder()
    df_clean["Location_Encoded"] = le.fit_transform(df_clean["Location"])

    # One-hot encoding of Location (useful for linear models)
    onehot = pd.get_dummies(df_clean["Location"], prefix="Loc")

    feature_cols_numeric = NUMERIC_COLUMNS + ["Floor_Ratio", "Location_Encoded"]
    X = pd.concat([df_clean[feature_cols_numeric], onehot], axis=1)
    y = df_clean["Price"]

    return df_clean, X, y, le, list(onehot.columns)


# ------------------------------------------------------------------
# 4. Split + scale
# ------------------------------------------------------------------
def split_and_scale(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ------------------------------------------------------------------
# 5 & 6. Train + evaluate all models
# ------------------------------------------------------------------
def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=12),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, max_depth=15, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, random_state=42, learning_rate=0.08, max_depth=4
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=250, random_state=42, learning_rate=0.08,
            max_depth=5, verbosity=0
        )
    return models


def evaluate_model(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def train_and_compare(models, X_train, X_test, y_train, y_test,
                       X_train_scaled, X_test_scaled):
    """
    Linear Regression uses scaled features; tree-based ensemble models
    use raw (unscaled) features, which is standard practice since trees
    are scale-invariant.
    """
    results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining: {name} ...")
        if name == "Linear Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        metrics = evaluate_model(y_test, preds)
        results[name] = metrics
        fitted_models[name] = model

        print(f"  MAE:  {metrics['MAE']:,.2f}")
        print(f"  MSE:  {metrics['MSE']:,.2f}")
        print(f"  RMSE: {metrics['RMSE']:,.2f}")
        print(f"  R2:   {metrics['R2']:.4f}")

    return results, fitted_models


# ------------------------------------------------------------------
# 7. Select best model
# ------------------------------------------------------------------
def select_best_model(results: dict, fitted_models: dict):
    best_name = max(results, key=lambda name: results[name]["R2"])
    best_model = fitted_models[best_name]
    print(f"\n>>> Best model selected: {best_name} (R2 = {results[best_name]['R2']:.4f})")
    return best_name, best_model


# ------------------------------------------------------------------
# 9. Visualizations
# ------------------------------------------------------------------
def generate_visualizations(df: pd.DataFrame, results: dict, best_model_name: str,
                             fitted_models: dict, X_train, X_test, y_test,
                             onehot_cols, feature_names_full):
    ensure_dir(IMAGES_DIR)

    # 1. Price distribution histogram
    plt.figure()
    sns.histplot(df["Price"] / 1e5, bins=40, kde=True, color="#2563eb")
    plt.title("Price Distribution (in Lakhs)")
    plt.xlabel("Price (Lakh INR)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/price_distribution.png", dpi=120)
    plt.close()

    # 2. Price vs Area scatter plot
    plt.figure()
    sns.scatterplot(data=df, x="Area_SqFt", y="Price", hue="BHK",
                     palette="viridis", alpha=0.6)
    plt.title("Price vs Area (SqFt)")
    plt.xlabel("Area (SqFt)")
    plt.ylabel("Price (INR)")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/price_vs_area.png", dpi=120)
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(9, 7))
    corr_cols = NUMERIC_COLUMNS + ["Price"]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/correlation_heatmap.png", dpi=120)
    plt.close()

    # 4. Location-wise average price
    plt.figure(figsize=(10, 7))
    loc_avg = df.groupby("Location")["Price"].mean().sort_values() / 1e5
    loc_avg.plot(kind="barh", color="#059669")
    plt.title("Location-wise Average Price (Lakhs)")
    plt.xlabel("Average Price (Lakh INR)")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/location_avg_price.png", dpi=120)
    plt.close()

    # 5. BHK distribution
    plt.figure()
    sns.countplot(data=df, x="BHK", hue="BHK", palette="magma", legend=False)
    plt.title("BHK Distribution")
    plt.xlabel("BHK")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/bhk_distribution.png", dpi=120)
    plt.close()

    # 6. Model comparison (R2 scores)
    plt.figure()
    names = list(results.keys())
    r2_vals = [results[n]["R2"] for n in names]
    colors = ["#059669" if n == best_model_name else "#94a3b8" for n in names]
    plt.bar(names, r2_vals, color=colors)
    plt.title("Model Comparison - R2 Score")
    plt.ylabel("R2 Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/model_comparison.png", dpi=120)
    plt.close()

    # 7. Feature importance (best model, if tree-based)
    best_model = fitted_models[best_model_name]
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names_full)
        feat_imp = feat_imp.sort_values(ascending=False).head(12)
        plt.figure(figsize=(9, 7))
        feat_imp.sort_values().plot(kind="barh", color="#7c3aed")
        plt.title(f"Feature Importance ({best_model_name})")
        plt.tight_layout()
        plt.savefig(f"{IMAGES_DIR}/feature_importance.png", dpi=120)
        plt.close()

    print(f"\nVisualizations saved to '{IMAGES_DIR}/' folder.")


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def main():
    df_raw = load_data(DATA_PATH)
    df_clean, X, y, label_encoder, onehot_cols = preprocess(df_raw)

    (X_train, X_test, X_train_scaled, X_test_scaled,
     y_train, y_test, scaler) = split_and_scale(X, y)

    models = get_models()
    results, fitted_models = train_and_compare(
        models, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    )

    best_name, best_model = select_best_model(results, fitted_models)

    print("\n--- Final Model Comparison Table ---")
    results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    print(results_df.round(4))

    generate_visualizations(
        df_clean, results, best_name, fitted_models,
        X_train, X_test, y_test, onehot_cols, list(X.columns)
    )

    # Save model bundle: model + scaler + encoder + metadata
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_columns": list(X.columns),
        "onehot_columns": onehot_cols,
        "uses_scaled_input": best_name == "Linear Regression",
        "metrics": results[best_name],
        "all_results": results,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nModel bundle saved to '{MODEL_PATH}'")

    # Save cleaned dataset for use in the Streamlit app (locality averages etc.)
    df_clean.to_csv("dataset/cleaned_dataset.csv", index=False)
    print("Cleaned dataset saved to 'dataset/cleaned_dataset.csv'")


if __name__ == "__main__":
    main()
