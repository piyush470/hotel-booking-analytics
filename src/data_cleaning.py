"""
data_cleaning.py
----------------
Hotel Booking Analytics & Cancellation Prediction
Piyush Raikwar | B.Tech AI & Data Science

Performs all data cleaning steps on the raw hotel_bookings dataset.
Returns a cleaned DataFrame ready for EDA and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(path: str | Path = "data/raw/hotel_bookings.csv") -> pd.DataFrame:
    """Load the raw CSV without modifying it."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply all cleaning steps and return a cleaned copy.
    Original DataFrame is NOT modified.
    """
    df = df.copy()
    original_shape = df.shape

    if verbose:
        print(f"[Cleaning] Starting shape: {original_shape}")

    # ── 1. DUPLICATE ROWS ──────────────────────────────────────────────────
    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    if verbose:
        print(f"[Cleaning] Dropped {n_dupes} duplicate rows -> {df.shape[0]} rows remain")

    # ── 2. MISSING VALUES ──────────────────────────────────────────────────
    # children: 4 missing -> fill with 0 (no children, numeric context)
    n_children_na = df["children"].isna().sum()
    df["children"] = df["children"].fillna(0).astype(int)
    if verbose:
        print(f"[Cleaning] 'children' NaN filled with 0: {n_children_na} records")

    # country: 488 missing -> fill with 'Unknown'
    n_country_na = df["country"].isna().sum()
    df["country"] = df["country"].fillna("Unknown")
    if verbose:
        print(f"[Cleaning] 'country' NaN filled with 'Unknown': {n_country_na} records")

    # agent: 16340 missing -> represents no-agent bookings, fill with 0
    n_agent_na = df["agent"].isna().sum()
    df["agent"] = df["agent"].fillna(0).astype(int)
    if verbose:
        print(f"[Cleaning] 'agent' NaN filled with 0: {n_agent_na} records")

    # company: 112593 missing (~94%) -> fill with 0 (no corporate company)
    n_company_na = df["company"].isna().sum()
    df["company"] = df["company"].fillna(0).astype(int)
    if verbose:
        print(f"[Cleaning] 'company' NaN filled with 0: {n_company_na} records")

    # ── 3. INVALID / SUSPICIOUS VALUES ────────────────────────────────────
    # ADR < 0: set to NaN then median-fill (only 1 record)
    n_neg_adr = (df["adr"] < 0).sum()
    df.loc[df["adr"] < 0, "adr"] = np.nan
    df["adr"] = df["adr"].fillna(df["adr"].median())
    if verbose:
        print(f"[Cleaning] Negative ADR records corrected: {n_neg_adr}")

    # ADR > 5000: extreme outlier – cap at 5000 (1 record, value=5400)
    n_adr_extreme = (df["adr"] > 5000).sum()
    df.loc[df["adr"] > 5000, "adr"] = 5000
    if verbose:
        print(f"[Cleaning] Extreme ADR (>5000) capped: {n_adr_extreme}")

    # adults == 0 AND children == 0 AND babies == 0 -> truly zero-guest records
    zero_guests = ((df["adults"] == 0) & (df["children"] == 0) & (df["babies"] == 0))
    n_zero = zero_guests.sum()
    df = df[~zero_guests]
    if verbose:
        print(f"[Cleaning] Removed zero-guest records: {n_zero}")

    # Stays both 0 (no nights) after removing zero guests
    zero_stays = ((df["stays_in_weekend_nights"] == 0) & (df["stays_in_week_nights"] == 0))
    n_zero_stays = zero_stays.sum()
    # Keep these – they might be day-use or valid 0-night stays; flag only
    if verbose:
        print(f"[Cleaning] Zero-total-stay records (kept, flagged): {n_zero_stays}")

    # meal: 'Undefined' -> map to 'SC' (self-catering / no meal) as domain equivalent
    n_undefined_meal = (df["meal"] == "Undefined").sum()
    df["meal"] = df["meal"].replace("Undefined", "SC")
    if verbose:
        print(f"[Cleaning] 'meal' Undefined -> SC: {n_undefined_meal} records")

    # ── 4. DATA TYPES ──────────────────────────────────────────────────────
    # Convert arrival_date_month to integer via month map for numeric use
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["arrival_month_num"] = df["arrival_date_month"].map(month_map)

    # reservation_status_date -> datetime (kept but will be excluded from ML)
    df["reservation_status_date"] = pd.to_datetime(
        df["reservation_status_date"], errors="coerce"
    )

    if verbose:
        print(f"[Cleaning] Final shape after all cleaning: {df.shape}")

    return df


def save_cleaned_data(
    df: pd.DataFrame,
    path: str | Path = "data/processed/cleaned_hotel_bookings.csv"
) -> None:
    """Persist the cleaned DataFrame to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[Cleaning] Cleaned data saved to {path}")


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    save_cleaned_data(clean_df)
