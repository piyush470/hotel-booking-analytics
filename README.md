# Hotel Booking Analytics & Cancellation Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

A complete, end-to-end Data Science and Data Analytics project that analyses hotel booking patterns and predicts booking cancellations using machine learning. The project covers data cleaning, exploratory data analysis, feature engineering, machine learning, model evaluation, and an interactive Streamlit dashboard.

---

## Problem Statement

Hotels lose significant revenue to last-minute booking cancellations. Understanding the factors that drive cancellations enables hotels to:

- Adjust overbooking strategies to compensate for predicted cancellations
- Target at-risk customers with personalised retention offers
- Improve revenue forecasting accuracy

**Goal:** Analyse hotel booking data to discover booking, customer, pricing, and cancellation patterns, and develop a machine learning model that predicts whether a booking is likely to be cancelled.

---

## Objectives

1. Perform thorough Exploratory Data Analysis (EDA) on hotel booking data
2. Answer 24+ specific business questions across booking, customer, pricing, and cancellation analytics
3. Build and compare multiple machine learning models for cancellation prediction
4. Deploy an interactive Streamlit dashboard with live prediction capability

---

## Dataset

**Source:** [Hotel Booking Demand – Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

| Property | Value |
|----------|-------|
| File | `hotel_bookings.csv` |
| Raw rows | 119,390 |
| Raw columns | 32 |
| Cleaned rows | 87,230 |
| Date range | July 2015 – August 2017 |
| Hotel types | City Hotel, Resort Hotel |

**Important features:**
- `hotel` – Hotel type (City Hotel / Resort Hotel)
- `lead_time` – Days between booking and arrival
- `adr` – Average Daily Rate (price per night)
- `deposit_type` – No Deposit / Non Refund / Refundable
- `market_segment` – Booking channel (Online TA, Direct, Corporate, etc.)
- `customer_type` – Transient, Contract, Group, Transient-Party
- `previous_cancellations` – Guest's historical cancellations
- `total_of_special_requests` – Number of special requests made

**Target variable:** `is_canceled` (0 = Not Cancelled, 1 = Cancelled)

> **Note:** The raw CSV is not committed to this repository due to file size.  
> Download from Kaggle and place at `data/raw/hotel_bookings.csv`.

---

## Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| Data manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Dashboard | Streamlit |
| Model persistence | Joblib |
| Documentation | python-docx |
| Version control | Git / GitHub |

---

## Project Workflow

```
Dataset (hotel_bookings.csv)
        ↓
  Data Cleaning
  (duplicates, missing values, invalid records)
        ↓
  Exploratory Data Analysis
  (24+ business questions, 17 charts)
        ↓
  Feature Engineering
  (10 new features derived)
        ↓
  Machine Learning
  (4 models trained & compared)
        ↓
  Model Evaluation
  (Accuracy, Precision, Recall, F1, ROC-AUC)
        ↓
  Interactive Dashboard
  (Streamlit – 4 pages)
        ↓
  GitHub / Streamlit Cloud Deployment
```

---

## Key Analysis Findings

> All findings are observational/associative. Causal inference requires controlled experiments.

| Finding | Detail |
|---------|--------|
| Cancellation rate | ~27.5% of cleaned bookings are cancelled |
| Peak demand month | August (11,242 bookings) |
| Lowest demand month | January (4,685 bookings) |
| Lead time difference | Cancelled bookings average 106 days lead time vs 70 days for honoured |
| Deposit type | Non-refundable deposits show the highest cancellation rate |
| Repeat guests | Cancel at only ~7.7% vs ~28.3% for new guests |
| Special requests | More requests → lower cancellation probability |
| Top country | Portugal (PRT) dominates bookings |
| Top market segment | Online Travel Agency (Online TA) |

---

## Machine Learning

### Target Variable
`is_canceled` — Binary classification: 0 (Not Cancelled) / 1 (Cancelled)

### Data Leakage Prevention
The following columns are **excluded** to prevent leakage:
- `reservation_status` — directly encodes the outcome
- `reservation_status_date` — post-outcome date

### Models Compared

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **Gradient Boosting** | **0.8178** | **0.7365** | 0.5267 | 0.6141 | **0.8623** |
| Random Forest | 0.7484 | 0.5278 | **0.8140** | **0.6404** | 0.8577 |
| Decision Tree | 0.7076 | 0.4822 | 0.8444 | 0.6138 | 0.8387 |
| Logistic Regression | 0.7207 | 0.4954 | 0.7876 | 0.6082 | 0.8341 |

**Best model: Gradient Boosting** (ROC-AUC = 0.8623)

### Evaluation Metrics Explained
- **Accuracy** — Overall correctness (reliable because class imbalance is moderate)
- **Precision** — Of predicted cancellations, how many are actually cancelled
- **Recall** — Of actual cancellations, how many did the model catch
- **F1-score** — Harmonic mean of precision and recall
- **ROC-AUC** — Model's discrimination ability across all thresholds (primary metric)

---

## Dashboard

The Streamlit dashboard (`dashboard/app.py`) contains four pages:

| Page | Content |
|------|---------|
| Executive Overview | KPIs: total bookings, cancel rate, avg ADR, avg lead time, avg stay |
| Booking Analytics | Monthly trends, hotel comparison, customer types, market segments |
| Cancellation Analytics | Cancel by deposit type, customer type, segment, lead time, special requests |
| ML Prediction | Interactive form → cancellation prediction with probability |

Live Demo: **[ADD DEPLOYED URL AFTER DEPLOYMENT]**

---

## Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd hotel-booking-analytics

# 2. Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset from Kaggle and place it at:
#    data/raw/hotel_bookings.csv

# 5. Run the data pipeline (cleaning, training, saving models)
python run_full_pipeline.py
```

## Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Run the Full Pipeline

```bash
python run_full_pipeline.py
```

This will:
- Clean the raw data → `data/processed/cleaned_hotel_bookings.csv`
- Generate 17 EDA charts → `images/`
- Train 4 ML models
- Save the best model → `models/model.pkl`

---

## Project Structure

```
hotel-booking-analytics/
│
├── data/
│   ├── raw/
│   │   └── hotel_bookings.csv          ← Place Kaggle CSV here
│   └── processed/
│       └── cleaned_hotel_bookings.csv
│
├── notebooks/
│   └── Piyush_Hotel_Booking_Analytics.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   └── model_training.py
│
├── models/
│   ├── model.pkl                        ← Best trained model
│   ├── best_model_name.pkl
│   ├── feature_columns.pkl
│   └── model_comparison.csv
│
├── dashboard/
│   └── app.py                           ← Streamlit dashboard
│
├── images/
│   └── [17 EDA & ML charts]
│
├── run_full_pipeline.py                 ← End-to-end pipeline script
├── requirements.txt
├── README.md
├── Piyush_Hotel_Booking_Project_Report.docx
└── .gitignore
```

---

## Results

| Metric | Value |
|--------|-------|
| Best model | Gradient Boosting |
| ROC-AUC | 0.8623 |
| Accuracy | 81.78% |
| Precision (cancelled) | 73.65% |
| Recall (cancelled) | 52.67% |
| F1 (cancelled) | 61.41% |
| Training samples | 69,784 |
| Test samples | 17,446 |

**Top influential features:**
1. Deposit type (Non Refund)
2. Lead time
3. Arrival month (numeric)
4. Previous cancellations
5. Total special requests
6. Cancellation history rate

---

## Future Improvements

- **Hyperparameter tuning** — GridSearchCV/RandomizedSearchCV for further optimisation
- **SHAP explanations** — Per-prediction feature contribution visualisations
- **Class balancing** — SMOTE or adjusted class weights for improved recall
- **Temporal validation** — Time-based train/test split to simulate production deployment
- **FastAPI endpoint** — REST API for real-time prediction integration with PMS systems
- **Extended feature engineering** — Price per guest, booking-to-capacity ratio

---

## Deployment to Streamlit Community Cloud

1. Push the repository to GitHub (ensure `models/` and `data/processed/` are committed)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path: `dashboard/app.py`
5. Click Deploy

> Note: The `data/raw/hotel_bookings.csv` file may exceed GitHub's 100 MB limit.
> If so, commit only `data/processed/cleaned_hotel_bookings.csv` and ensure
> the dashboard loads from `data/processed/`.

---

## Git Commands

```bash
# Initialise and push to GitHub
git init
git add .
git commit -m "Initial commit: Hotel Booking Analytics & Cancellation Prediction"
git branch -M main
git remote add origin https://github.com/<your-username>/hotel-booking-analytics.git
git push -u origin main
```

---

## Author

**Piyush Raikwar**  
B.Tech – Artificial Intelligence & Data Science  

---

## License

This project is for educational and portfolio purposes.
This project is created under IBM Skillbuild Virtual Internship using IBM BOB IDE  
Dataset sourced from Kaggle under the original dataset license.
