"""
run_full_pipeline.py
--------------------
Runs the complete data pipeline: cleaning, EDA charts, feature engineering,
model training, evaluation, and saves all artefacts.
Run from project root: python run_full_pipeline.py
"""
import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from src.data_cleaning import load_raw_data, clean_data, save_cleaned_data
from src.feature_engineering import engineer_features
from src.preprocessing import build_preprocessor, get_X_y, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.model_training import (
    split_data, build_model_pipelines, evaluate_model,
    get_feature_importance, RANDOM_STATE
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, classification_report
)

PALETTE = sns.color_palette('Set2')
plt.rcParams.update({'figure.dpi': 100, 'axes.titlesize': 12, 'axes.labelsize': 10})
sns.set_style('whitegrid')
IMG = Path('images')
IMG.mkdir(exist_ok=True)

print("=" * 60)
print("HOTEL BOOKING ANALYTICS – FULL PIPELINE")
print("=" * 60)

# ── 1. LOAD & CLEAN ───────────────────────────────────────────
print("\n[1] Loading raw data...")
raw_df = load_raw_data('data/raw/hotel_bookings.csv')
print(f"    Raw shape: {raw_df.shape}")

print("\n[2] Cleaning data...")
clean_df = clean_data(raw_df, verbose=True)
save_cleaned_data(clean_df, 'data/processed/cleaned_hotel_bookings.csv')

# ── 2. EDA CHARTS ─────────────────────────────────────────────
print("\n[3] Generating EDA charts...")
month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']

# Cancellation distribution
counts = clean_df['is_canceled'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['Not Cancelled\n(0)', 'Cancelled\n(1)'], counts.values,
            color=[PALETTE[1], PALETTE[3]])
axes[0].set_title('Booking Cancellation Distribution')
axes[0].set_ylabel('Number of Bookings')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 200, f'{v:,}\n({v/len(clean_df)*100:.1f}%)', ha='center', fontsize=9)
axes[1].pie(counts.values, labels=['Not Cancelled', 'Cancelled'],
            autopct='%1.1f%%', colors=[PALETTE[1], PALETTE[3]], startangle=90)
axes[1].set_title('Cancellation Rate')
plt.tight_layout()
plt.savefig(IMG / '01_cancellation_distribution.png', bbox_inches='tight')
plt.close()

# Hotel type
hotel_counts = clean_df['hotel'].value_counts()
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(hotel_counts.index, hotel_counts.values, color=[PALETTE[0], PALETTE[2]])
ax.set_title('Number of Bookings by Hotel Type')
ax.set_ylabel('Number of Bookings')
for i, v in enumerate(hotel_counts.values):
    ax.text(i, v + 100, f'{v:,}', ha='center')
plt.tight_layout()
plt.savefig(IMG / '02_hotel_type_distribution.png', bbox_inches='tight')
plt.close()

# Monthly demand
monthly = clean_df.groupby('arrival_date_month').size().reindex(month_order)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(month_order, monthly.values, marker='o', linewidth=2, color=PALETTE[0])
ax.fill_between(month_order, monthly.values, alpha=0.2, color=PALETTE[0])
ax.set_title('Monthly Booking Demand')
ax.set_ylabel('Number of Bookings')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(IMG / '03_monthly_demand.png', bbox_inches='tight')
plt.close()
print(f"    Peak month: {monthly.idxmax()} ({monthly.max():,})")
print(f"    Lowest month: {monthly.idxmin()} ({monthly.min():,})")

# Monthly cancellation rate
monthly_cancel = clean_df.groupby('arrival_date_month')['is_canceled'].mean().reindex(month_order) * 100
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(month_order, monthly_cancel.values, color=PALETTE[3], alpha=0.8)
ax.set_title('Monthly Cancellation Rate (%)')
ax.set_ylabel('Cancellation Rate (%)')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(IMG / '04_monthly_cancellation_rate.png', bbox_inches='tight')
plt.close()

# Lead time
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(clean_df['lead_time'], bins=50, color=PALETTE[0], edgecolor='white')
axes[0].set_title('Lead Time Distribution')
axes[0].set_xlabel('Lead Time (days)')
axes[0].set_ylabel('Count')
cancelled = clean_df[clean_df['is_canceled']==1]['lead_time']
not_cancelled = clean_df[clean_df['is_canceled']==0]['lead_time']
axes[1].hist(not_cancelled, bins=50, alpha=0.6, label='Not Cancelled', color=PALETTE[1])
axes[1].hist(cancelled, bins=50, alpha=0.6, label='Cancelled', color=PALETTE[3])
axes[1].set_title('Lead Time: Cancelled vs Not Cancelled')
axes[1].set_xlabel('Lead Time (days)')
axes[1].legend()
plt.tight_layout()
plt.savefig(IMG / '05_lead_time_distribution.png', bbox_inches='tight')
plt.close()
print(f"    Avg lead time (not cancelled): {not_cancelled.mean():.0f} days")
print(f"    Avg lead time (cancelled):     {cancelled.mean():.0f} days")

# ADR analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
hotel_adr = clean_df.groupby('hotel')['adr'].mean()
axes[0].bar(hotel_adr.index, hotel_adr.values, color=[PALETTE[0], PALETTE[2]])
axes[0].set_title('Average Daily Rate by Hotel Type')
axes[0].set_ylabel('Average ADR (EUR)')
for i, v in enumerate(hotel_adr.values):
    axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center')
monthly_adr = clean_df.groupby('arrival_date_month')['adr'].mean().reindex(month_order)
axes[1].plot(month_order, monthly_adr.values, marker='o', linewidth=2, color=PALETTE[2])
axes[1].fill_between(month_order, monthly_adr.values, alpha=0.2, color=PALETTE[2])
axes[1].set_title('Average Daily Rate by Month')
axes[1].set_ylabel('ADR (EUR)')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30)
plt.tight_layout()
plt.savefig(IMG / '06_adr_analysis.png', bbox_inches='tight')
plt.close()
print(f"    Overall avg ADR: EUR {clean_df['adr'].mean():.2f}")

# Market segment
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
seg_counts = clean_df['market_segment'].value_counts()
axes[0].barh(seg_counts.index, seg_counts.values, color=PALETTE)
axes[0].set_title('Bookings by Market Segment')
axes[0].set_xlabel('Number of Bookings')
seg_cancel = clean_df.groupby('market_segment')['is_canceled'].mean().sort_values(ascending=False) * 100
axes[1].barh(seg_cancel.index, seg_cancel.values, color=PALETTE[3])
axes[1].set_title('Cancellation Rate by Market Segment')
axes[1].set_xlabel('Cancellation Rate (%)')
axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
plt.tight_layout()
plt.savefig(IMG / '07_market_segment.png', bbox_inches='tight')
plt.close()

# Deposit type
deposit_cancel = clean_df.groupby('deposit_type')['is_canceled'].mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(deposit_cancel.index, deposit_cancel.values, color=[PALETTE[3], PALETTE[1], PALETTE[0]])
ax.set_title('Cancellation Rate by Deposit Type')
ax.set_ylabel('Cancellation Rate (%)')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
for i, v in enumerate(deposit_cancel.values):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.savefig(IMG / '08_deposit_type_cancellation.png', bbox_inches='tight')
plt.close()

# Customer type
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cust_counts = clean_df['customer_type'].value_counts()
axes[0].pie(cust_counts.values, labels=cust_counts.index,
            autopct='%1.1f%%', colors=PALETTE, startangle=90)
axes[0].set_title('Customer Type Distribution')
cust_cancel = clean_df.groupby('customer_type')['is_canceled'].mean().sort_values(ascending=False) * 100
axes[1].bar(cust_cancel.index, cust_cancel.values, color=PALETTE)
axes[1].set_title('Cancellation Rate by Customer Type')
axes[1].set_ylabel('Cancellation Rate (%)')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
plt.tight_layout()
plt.savefig(IMG / '09_customer_type.png', bbox_inches='tight')
plt.close()

# Repeat guest
repeat_cancel = clean_df.groupby('is_repeated_guest')['is_canceled'].mean() * 100
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(['New Guest', 'Repeat Guest'], repeat_cancel.values, color=[PALETTE[3], PALETTE[1]])
ax.set_title('Cancellation Rate: New vs Repeat Guests')
ax.set_ylabel('Cancellation Rate (%)')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
for i, v in enumerate(repeat_cancel.values):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.savefig(IMG / '10_repeat_guest_cancellation.png', bbox_inches='tight')
plt.close()
print(f"    Repeat guest cancellation: {repeat_cancel[1]:.1f}%")
print(f"    New guest cancellation:    {repeat_cancel[0]:.1f}%")

# Correlation heatmap
num_cols = ['lead_time','stays_in_weekend_nights','stays_in_week_nights',
            'adults','children','babies','is_repeated_guest','previous_cancellations',
            'previous_bookings_not_canceled','booking_changes','days_in_waiting_list',
            'adr','required_car_parking_spaces','total_of_special_requests','is_canceled']
corr = clean_df[num_cols].corr()
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5)
ax.set_title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig(IMG / '11_correlation_heatmap.png', bbox_inches='tight')
plt.close()

# Top countries
top_countries = clean_df[clean_df['country'] != 'Unknown']['country'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(top_countries.index[::-1], top_countries.values[::-1], color=PALETTE[0])
ax.set_title('Top 10 Countries by Number of Bookings')
ax.set_xlabel('Number of Bookings')
plt.tight_layout()
plt.savefig(IMG / '12_top_countries.png', bbox_inches='tight')
plt.close()

# Length of stay
stay_data = clean_df.copy()
stay_data['total_nights'] = stay_data['stays_in_weekend_nights'] + stay_data['stays_in_week_nights']
stay_data = stay_data[stay_data['total_nights'] <= 20]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(stay_data['total_nights'], bins=20, color=PALETTE[0], edgecolor='white')
axes[0].set_title('Total Length of Stay Distribution')
axes[0].set_xlabel('Total Nights')
axes[0].set_ylabel('Count')
stay_cancel = stay_data.groupby('total_nights')['is_canceled'].mean() * 100
axes[1].plot(stay_cancel.index, stay_cancel.values, color=PALETTE[3], marker='o')
axes[1].set_title('Cancellation Rate by Length of Stay')
axes[1].set_xlabel('Total Nights')
axes[1].set_ylabel('Cancellation Rate (%)')
plt.tight_layout()
plt.savefig(IMG / '13_length_of_stay.png', bbox_inches='tight')
plt.close()
print(f"    Avg total stay: {stay_data['total_nights'].mean():.1f} nights")

print("\n    All EDA charts saved to images/")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────
print("\n[4] Feature engineering...")
fe_df = engineer_features(clean_df, verbose=True)

# ── 4. ML TRAINING ────────────────────────────────────────────
print("\n[5] Machine learning training...")
X, y = get_X_y(fe_df)
X_train, X_test, y_train, y_test = split_data(X, y)
print(f"    Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

preprocessor = build_preprocessor()
pipelines = build_model_pipelines(preprocessor)
results = {}

for name, pipeline in pipelines.items():
    print(f"    Training {name}...", end=' ', flush=True)
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    results[name] = metrics
    print(f"ROC-AUC={metrics['ROC-AUC']:.4f}")

results_df = pd.DataFrame(results).T.round(4)
results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
results_df = results_df.sort_values('ROC-AUC', ascending=False)
print("\n=== MODEL COMPARISON ===")
print(results_df.to_string())

# ── 5. ML CHARTS ──────────────────────────────────────────────
print("\n[6] Generating ML evaluation charts...")

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for idx, (name, pipeline) in enumerate(pipelines.items()):
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Not Cancelled', 'Cancelled'])
    disp.plot(ax=axes[idx], colorbar=False, cmap='Blues')
    axes[idx].set_title(name)
plt.suptitle('Confusion Matrices', fontsize=14)
plt.tight_layout()
plt.savefig(IMG / '14_confusion_matrices.png', bbox_inches='tight')
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(9, 7))
for name, pipeline in pipelines.items():
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax.set_title('ROC Curves – All Models')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(IMG / '15_roc_curves.png', bbox_inches='tight')
plt.close()

# Model comparison bar
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.2
fig, ax = plt.subplots(figsize=(13, 6))
for i, (name, row) in enumerate(results_df.iterrows()):
    ax.bar(x + i * width, [row[m] for m in metrics], width, label=name, alpha=0.8)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.set_title('Model Metric Comparison')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(IMG / '16_model_comparison.png', bbox_inches='tight')
plt.close()

# Feature importance
best_name = results_df.index[0]
best_pipeline = pipelines[best_name]
importance_df = get_feature_importance(best_pipeline, list(X.columns))
fig, ax = plt.subplots(figsize=(10, 8))
top_n = importance_df.head(15)
ax.barh(top_n['feature'].values[::-1], top_n['importance'].values[::-1], color=PALETTE[0])
ax.set_title(f'Top 15 Feature Importances – {best_name}')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(IMG / '17_feature_importance.png', bbox_inches='tight')
plt.close()
print("    ML charts saved.")

# ── 6. SAVE MODELS ────────────────────────────────────────────
print("\n[7] Saving models...")
Path('models').mkdir(exist_ok=True)
joblib.dump(best_pipeline, 'models/model.pkl')
joblib.dump(best_name, 'models/best_model_name.pkl')
joblib.dump(list(X.columns), 'models/feature_columns.pkl')
print(f"    Best model ({best_name}) saved to models/model.pkl")

# Save results summary
results_df.to_csv('models/model_comparison.csv')
print("    Model comparison saved to models/model_comparison.csv")

# Print final classification report
y_pred_best = best_pipeline.predict(X_test)
print(f"\n=== Classification Report: {best_name} ===")
print(classification_report(y_test, y_pred_best, target_names=['Not Cancelled', 'Cancelled']))

print("\n" + "="*60)
print("PIPELINE COMPLETE - All artefacts generated successfully.")
print("="*60)
