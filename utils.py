"""
utils.py
--------
Shared helper functions for the House Price Predictor project.

Used by both train_model.py (training pipeline) and app.py (Streamlit UI).

Author: House Price Predictor Project
Python: 3.14.6
"""

import os
import numpy as np
import pandas as pd


LOCATIONS = [
    "Gachibowli", "Madhapur", "Kondapur", "Kukatpally", "Miyapur",
    "Hitech City", "Financial District", "Jubilee Hills", "Banjara Hills",
    "Manikonda", "Tellapur", "Nallagandla", "Begumpet", "Ameerpet",
    "Uppal", "LB Nagar", "Shaikpet", "Kompally", "Nizampet", "Bowenpally",
]

FEATURE_COLUMNS = [
    "Area_SqFt", "BHK", "Bathrooms", "Parking",
    "Property_Age", "Floor_Number", "Total_Floors", "Location",
]

NUMERIC_COLUMNS = [
    "Area_SqFt", "BHK", "Bathrooms", "Parking",
    "Property_Age", "Floor_Number", "Total_Floors",
]


def format_inr(amount: float) -> str:
    """
    Format a number as an Indian Rupee string with the Indian numbering
    system (lakh / crore comma placement).

    Example:
        format_inr(8250000) -> '₹ 82,50,000'
    """
    try:
        amount = int(round(amount))
    except (ValueError, TypeError):
        return "₹ 0"

    is_negative = amount < 0
    amount = abs(amount)

    s = str(amount)
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        formatted = ",".join(parts) + "," + last3

    result = f"₹ {formatted}"
    return f"-{result}" if is_negative else result


def remove_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame column using the IQR (Interquartile
    Range) method.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform full data-cleaning steps on the raw dataset:
      1. Drop duplicate rows
      2. Handle missing values
      3. Remove statistical outliers on Price
      4. Recompute Price_Per_SqFt to keep it consistent
    """
    df = df.copy()

    # 1. Remove duplicates
    df = df.drop_duplicates()

    # 2. Handle missing values
    #    - Numeric columns: fill with median (robust to outliers)
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    df = df.dropna(subset=["Location", "Price"])

    # 3. Remove outliers on Price
    df = remove_outliers_iqr(df, "Price", factor=1.5)

    # 4. Keep Price_Per_SqFt consistent with cleaned Price/Area
    df["Price_Per_SqFt"] = (df["Price"] / df["Area_SqFt"]).round(1)

    df = df.reset_index(drop=True)
    return df


def get_locality_avg_price(df: pd.DataFrame, location: str) -> float:
    """Return the average price for a given locality."""
    subset = df[df["Location"] == location]
    if subset.empty:
        return float(df["Price"].mean())
    return float(subset["Price"].mean())


def get_locality_avg_price_per_sqft(df: pd.DataFrame, location: str) -> float:
    """Return the average price per sqft for a given locality."""
    subset = df[df["Location"] == location]
    if subset.empty:
        return float(df["Price_Per_SqFt"].mean())
    return float(subset["Price_Per_SqFt"].mean())


def investment_suggestion(predicted_price: float, locality_avg_price: float) -> str:
    """
    Compare the predicted price against the locality average and return
    a simple investment classification.
    """
    if locality_avg_price <= 0:
        return "Average Investment"

    ratio = predicted_price / locality_avg_price

    if ratio <= 0.9:
        return "Good Investment"
    elif ratio <= 1.15:
        return "Average Investment"
    else:
        return "Premium Property"


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    if not os.path.exists(path):
        os.makedirs(path)
