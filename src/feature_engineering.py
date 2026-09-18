"""
feature_engineering.py
-----------------------
Hotel Booking Analytics & Cancellation Prediction
Piyush Raikwar | B.Tech AI & Data Science

Creates derived features from the cleaned hotel bookings dataset.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add engineered features to a cleaned DataFrame copy.
    Explain each feature to maintain traceability.
    """
    df = df.copy()

    # ── total_stay_nights ─────────────────────────────────────────────────
    # Total length of stay combining weekend and weekday nights.
    df["total_stay_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    if verbose:
        print("[FE] total_stay_nights: weekend + weekday nights")

    # ── total_guests ──────────────────────────────────────────────────────
    # Total number of guests: adults + children + babies.
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    if verbose:
        print("[FE] total_guests: adults + children + babies")

    # ── is_family ─────────────────────────────────────────────────────────
    # Binary flag: 1 if booking includes at least one child or baby.
    df["is_family"] = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)
    if verbose:
        print("[FE] is_family: 1 if children>0 or babies>0")

    # ── is_weekend_booking ────────────────────────────────────────────────
    # Binary flag: 1 if the booking includes at least one weekend night.
    df["is_weekend_booking"] = (df["stays_in_weekend_nights"] > 0).astype(int)
    if verbose:
        print("[FE] is_weekend_booking: 1 if stays_in_weekend_nights > 0")

    # ── arrival_month_num ─────────────────────────────────────────────────
    # Numeric encoding of arrival month (already added in cleaning if not present).
    if "arrival_month_num" not in df.columns:
        month_map = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        df["arrival_month_num"] = df["arrival_date_month"].map(month_map)
        if verbose:
            print("[FE] arrival_month_num: numeric month")

    # ── has_company ───────────────────────────────────────────────────────
    # Binary flag: 1 if booking is associated with a corporate company.
    df["has_company"] = (df["company"] > 0).astype(int)
    if verbose:
        print("[FE] has_company: 1 if company != 0")

    # ── has_agent ─────────────────────────────────────────────────────────
    # Binary flag: 1 if booking was made through a travel agent.
    df["has_agent"] = (df["agent"] > 0).astype(int)
    if verbose:
        print("[FE] has_agent: 1 if agent != 0")

    # ── cancellation_history_rate ─────────────────────────────────────────
    # Ratio of previous cancellations to total previous bookings.
    # Indicates the guest's historical cancellation tendency.
    total_prev = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["cancellation_history_rate"] = np.where(
        total_prev > 0,
        df["previous_cancellations"] / total_prev,
        0.0
    )
    if verbose:
        print("[FE] cancellation_history_rate: prev_cancellations / total_prev_bookings")

    # ── room_type_match ───────────────────────────────────────────────────
    # Binary flag: 1 if the assigned room type matches the reserved room type.
    # A mismatch may indicate dissatisfaction → potentially higher cancellation.
    df["room_type_match"] = (
        df["reserved_room_type"] == df["assigned_room_type"]
    ).astype(int)
    if verbose:
        print("[FE] room_type_match: 1 if reserved == assigned room type")

    # ── lead_time_bin ─────────────────────────────────────────────────────
    # Ordinal binning of lead time into categories.
    df["lead_time_bin"] = pd.cut(
        df["lead_time"],
        bins=[-1, 7, 30, 90, 180, 737],
        labels=["same_week", "short", "medium", "long", "very_long"]
    ).astype(str)
    if verbose:
        print("[FE] lead_time_bin: categorical lead time bucket")

    # ── adr_per_night ─────────────────────────────────────────────────────
    # ADR per night: same as ADR (ADR is already per night by definition,
    # but we create a safe version guarding against 0-night stays).
    df["adr_per_night"] = np.where(
        df["total_stay_nights"] > 0,
        df["adr"],
        0.0
    )
    if verbose:
        print("[FE] adr_per_night: ADR (with 0-night guard)")

    if verbose:
        print(f"[FE] Feature engineering complete. New shape: {df.shape}")

    return df


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_cleaning import load_raw_data, clean_data

    raw = load_raw_data()
    clean = clean_data(raw)
    fe = engineer_features(clean)
    print(fe.head())
