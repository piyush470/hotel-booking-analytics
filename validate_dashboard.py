"""
validate_dashboard.py - Tests dashboard model loading and prediction without Streamlit
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import joblib

DATA_CLEANED = Path('data/processed/cleaned_hotel_bookings.csv')
MODEL_PATH   = Path('models/model.pkl')
FEAT_PATH    = Path('models/feature_columns.pkl')

df = pd.read_csv(DATA_CLEANED)
model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEAT_PATH)

print(f'Data loaded: {df.shape}')
print(f'Model type: {type(model).__name__}')
print(f'Feature columns count: {len(feature_cols)}')

MONTH_ORDER = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
month_map = {m: i+1 for i, m in enumerate(MONTH_ORDER)}

test_input = {
    'lead_time': 120, 'stays_in_weekend_nights': 2, 'stays_in_week_nights': 3,
    'adults': 2, 'children': 0, 'babies': 0, 'is_repeated_guest': 0,
    'previous_cancellations': 0, 'previous_bookings_not_canceled': 0,
    'booking_changes': 0, 'days_in_waiting_list': 0, 'adr': 95.0,
    'required_car_parking_spaces': 0, 'total_of_special_requests': 1,
    'total_stay_nights': 5, 'total_guests': 2, 'is_family': 0,
    'is_weekend_booking': 1, 'arrival_month_num': 8, 'has_company': 0,
    'has_agent': 0, 'cancellation_history_rate': 0.0, 'room_type_match': 1,
    'adr_per_night': 95.0, 'hotel': 'City Hotel', 'meal': 'BB',
    'market_segment': 'Online TA', 'distribution_channel': 'TA/TO',
    'reserved_room_type': 'A', 'assigned_room_type': 'A',
    'deposit_type': 'No Deposit', 'customer_type': 'Transient',
    'lead_time_bin': 'long',
}

input_df = pd.DataFrame([test_input])
for col in feature_cols:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_cols]

pred = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1] * 100
label = "Cancelled" if pred == 1 else "Not Cancelled"
print(f'Test prediction: {label}')
print(f'Cancellation probability: {prob:.1f}%')
print('Dashboard model validation: PASS')
