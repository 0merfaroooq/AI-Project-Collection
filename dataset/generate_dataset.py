"""
generate_dataset.py
--------------------
Generates a realistic SYNTHETIC dataset of residential house prices
in Hyderabad, Telangana, India.

This script is run once to produce dataset/hyderabad_house_prices.csv,
which is then used by train_model.py.

Author: House Price Predictor Project
"""

import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# ------------------------------------------------------------------
# Locality configuration
# Each locality has a base price per sqft (INR) and a price volatility
# band, roughly reflecting real Hyderabad real-estate tiers as of 2025.
# ------------------------------------------------------------------
LOCALITY_CONFIG = {
    "Jubilee Hills":       {"base_price_sqft": 14500, "tier": "Premium"},
    "Banjara Hills":       {"base_price_sqft": 14000, "tier": "Premium"},
    "Financial District":  {"base_price_sqft": 9500,  "tier": "Premium"},
    "Hitech City":         {"base_price_sqft": 9000,  "tier": "Premium"},
    "Gachibowli":          {"base_price_sqft": 8500,  "tier": "High"},
    "Madhapur":            {"base_price_sqft": 8200,  "tier": "High"},
    "Kondapur":            {"base_price_sqft": 7200,  "tier": "High"},
    "Tellapur":            {"base_price_sqft": 6800,  "tier": "High"},
    "Nallagandla":         {"base_price_sqft": 6500,  "tier": "High"},
    "Manikonda":           {"base_price_sqft": 6200,  "tier": "Mid"},
    "Kukatpally":          {"base_price_sqft": 6000,  "tier": "Mid"},
    "Miyapur":             {"base_price_sqft": 5500,  "tier": "Mid"},
    "Nizampet":            {"base_price_sqft": 5300,  "tier": "Mid"},
    "Kompally":            {"base_price_sqft": 5000,  "tier": "Mid"},
    "Begumpet":            {"base_price_sqft": 7500,  "tier": "High"},
    "Ameerpet":            {"base_price_sqft": 6300,  "tier": "Mid"},
    "Bowenpally":          {"base_price_sqft": 5200,  "tier": "Mid"},
    "Shaikpet":            {"base_price_sqft": 6900,  "tier": "High"},
    "Uppal":                {"base_price_sqft": 4500,  "tier": "Affordable"},
    "LB Nagar":            {"base_price_sqft": 4300,  "tier": "Affordable"},
}

LOCALITIES = list(LOCALITY_CONFIG.keys())
N_SAMPLES = 3000


def generate_row():
    location = np.random.choice(LOCALITIES)
    config = LOCALITY_CONFIG[location]
    base_price_sqft = config["base_price_sqft"]

    # --- Property characteristics ---
    bhk = np.random.choice([1, 2, 3, 4, 5], p=[0.08, 0.35, 0.35, 0.17, 0.05])

    # Area correlates with BHK, with realistic noise
    area_base = {1: 650, 2: 1150, 3: 1650, 4: 2300, 5: 3200}[bhk]
    area = max(400, np.random.normal(area_base, area_base * 0.12))

    bathrooms = min(bhk + np.random.choice([0, 1], p=[0.6, 0.4]), 6)
    bathrooms = max(1, bathrooms)

    parking = np.random.choice([0, 1, 2, 3], p=[0.10, 0.55, 0.28, 0.07])

    property_age = np.random.choice(
        [0, 1, 2, 3, 5, 7, 10, 15, 20, 25],
        p=[0.10, 0.12, 0.12, 0.12, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04],
    )

    total_floors = np.random.choice(range(2, 26))
    floor_number = np.random.randint(0, total_floors + 1)

    # --- Price per sqft calculation ---
    price_sqft = base_price_sqft

    # Newer properties cost more per sqft
    price_sqft *= (1 - (property_age * 0.008))

    # Higher floors in high-rises get a small premium (view factor)
    if total_floors > 10:
        price_sqft *= (1 + (floor_number / total_floors) * 0.06)

    # More bathrooms/parking = slightly higher value density
    price_sqft *= (1 + (parking * 0.015))
    price_sqft *= (1 + (max(0, bathrooms - bhk) * 0.01))

    # Random market noise
    price_sqft *= np.random.normal(1.0, 0.07)
    price_sqft = max(2500, price_sqft)

    price = price_sqft * area

    return {
        "Location": location,
        "Area_SqFt": round(area, 1),
        "BHK": int(bhk),
        "Bathrooms": int(bathrooms),
        "Parking": int(parking),
        "Property_Age": int(property_age),
        "Floor_Number": int(floor_number),
        "Total_Floors": int(total_floors),
        "Price": round(price, 0),
        "Price_Per_SqFt": round(price_sqft, 1),
    }


def main():
    rows = [generate_row() for _ in range(N_SAMPLES)]
    df = pd.DataFrame(rows)

    # Introduce a small, realistic amount of missing data (to demonstrate
    # data-cleaning steps in train_model.py)
    missing_idx = np.random.choice(df.index, size=int(0.015 * len(df)), replace=False)
    df.loc[missing_idx, "Parking"] = np.nan

    # Introduce a handful of duplicate rows (to demonstrate deduplication)
    dup_rows = df.sample(15, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Introduce a few extreme outliers (to demonstrate outlier handling)
    outlier_idx = df.sample(6, random_state=2).index
    df.loc[outlier_idx, "Price"] = df.loc[outlier_idx, "Price"] * np.random.uniform(3.5, 5, size=6)

    df.to_csv("dataset/hyderabad_house_prices.csv", index=False)
    print(f"Dataset generated: {len(df)} rows -> dataset/hyderabad_house_prices.csv")
    print(df.head())


if __name__ == "__main__":
    main()
