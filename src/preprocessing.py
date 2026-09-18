"""
preprocessing.py
----------------
Hotel Booking Analytics & Cancellation Prediction
Piyush Raikwar | B.Tech AI & Data Science

Builds the scikit-learn preprocessing pipeline and defines the
feature sets used for model training.

DATA LEAKAGE PREVENTION:
  The following columns are EXCLUDED because they describe the
  *outcome* of a reservation, not information available at booking time:
    - reservation_status      → directly encodes cancellation
    - reservation_status_date → post-outcome date
    - is_canceled             → the target itself
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ── LEAKAGE-PRONE COLUMNS (excluded from all ML features) ─────────────────
LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]
TARGET_COL = "is_canceled"

# ── IDENTIFIER / ADMINISTRATIVE COLUMNS (not predictive) ──────────────────
# arrival_date_month is superseded by arrival_month_num (numeric)
# reservation_status_date already excluded above
DROP_FOR_ML = [
    "arrival_date_month",       # replaced by arrival_month_num
    "arrival_date_week_number", # granular week number; year+month sufficient
    "arrival_date_day_of_month",# too granular; not predictive at booking time
    "arrival_date_year",        # data spans 2015-2017; year not generalisable
]

# ── NUMERICAL FEATURES ────────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    # engineered
    "total_stay_nights",
    "total_guests",
    "is_family",
    "is_weekend_booking",
    "arrival_month_num",
    "has_company",
    "has_agent",
    "cancellation_history_rate",
    "room_type_match",
    "adr_per_night",
]

# ── CATEGORICAL FEATURES ──────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "lead_time_bin",
]


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer with:
      - Numerical: median imputation → standard scaling
      - Categorical: constant imputation → one-hot encoding (drop first to
        avoid multicollinearity)
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_X_y(df: pd.DataFrame):
    """
    Extract feature matrix X and target vector y from a cleaned + engineered df.
    Drops leakage columns and the target from X.
    """
    all_exclude = LEAKAGE_COLS + DROP_FOR_ML + [TARGET_COL]
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    # Keep only feature cols that exist in df
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[TARGET_COL]
    return X, y


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_cleaning import load_raw_data, clean_data
    from src.feature_engineering import engineer_features

    raw = load_raw_data()
    clean = clean_data(raw)
    fe = engineer_features(clean)
    X, y = get_X_y(fe)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Features:", list(X.columns))
