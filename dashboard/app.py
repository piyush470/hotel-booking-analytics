"""
dashboard/app.py
----------------
Hotel Booking Analytics & Cancellation Prediction
Interactive Streamlit Dashboard
Author: Piyush Raikwar | B.Tech AI & Data Science
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import streamlit as st
import joblib

# ── PATH RESOLUTION ───────────────────────────────────────────────────────────
# Works whether run from project root (streamlit run dashboard/app.py)
# or from within the dashboard/ folder.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CLEANED = BASE_DIR / "data" / "processed" / "cleaned_hotel_bookings.csv"
MODEL_PATH   = BASE_DIR / "models" / "model.pkl"
FEAT_PATH    = BASE_DIR / "models" / "feature_columns.pkl"

# ── STREAMLIT CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Booking Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#4CAF50", "#F44336", "#2196F3", "#FF9800", "#9C27B0",
           "#00BCD4", "#E91E63", "#8BC34A"]

MONTH_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# ── DATA & MODEL LOADERS ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_CLEANED)

@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEAT_PATH)
    return model, feature_cols

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hotel-building.png", width=80)
    st.title("Hotel Booking\nAnalytics")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["🏠 Executive Overview",
         "📊 Booking Analytics",
         "❌ Cancellation Analytics",
         "🤖 ML Prediction"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("**Author:** Piyush Raikwar")
    st.caption("B.Tech – AI & Data Science")
    st.caption("Dataset: Hotel Booking Demand")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df = load_data()

# ── PAGE 1: EXECUTIVE OVERVIEW ────────────────────────────────────────────────
if page == "🏠 Executive Overview":
    st.title("🏨 Hotel Booking Analytics & Cancellation Prediction")
    st.markdown("**Executive Overview — Key Performance Indicators**")
    st.markdown("---")

    total_bookings     = len(df)
    cancelled          = int(df["is_canceled"].sum())
    cancel_rate        = df["is_canceled"].mean() * 100
    avg_adr            = df["adr"].mean()
    avg_lead_time      = df["lead_time"].mean()
    avg_stay           = (df["stays_in_weekend_nights"] + df["stays_in_week_nights"]).mean()
    city_pct           = (df["hotel"] == "City Hotel").mean() * 100
    repeat_pct         = df["is_repeated_guest"].mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bookings", f"{total_bookings:,}")
    col2.metric("Cancellation Rate", f"{cancel_rate:.1f}%")
    col3.metric("Cancelled Bookings", f"{cancelled:,}")
    col4.metric("Average ADR (EUR)", f"{avg_adr:.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Lead Time (days)", f"{avg_lead_time:.0f}")
    col6.metric("Avg Stay (nights)", f"{avg_stay:.1f}")
    col7.metric("City Hotel Share", f"{city_pct:.1f}%")
    col8.metric("Repeat Guests", f"{repeat_pct:.1f}%")

    st.markdown("---")
    st.subheader("Cancellation Overview")

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df["is_canceled"].value_counts()
        ax.pie(counts.values,
               labels=["Not Cancelled", "Cancelled"],
               autopct="%1.1f%%",
               colors=[PALETTE[0], PALETTE[1]],
               startangle=90,
               textprops={"fontsize": 10})
        ax.set_title("Cancellation Distribution")
        st.pyplot(fig)
        plt.close()

    with col_b:
        hotel_cancel = df.groupby("hotel")["is_canceled"].agg(["mean", "count"])
        hotel_cancel["mean"] *= 100
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(hotel_cancel.index, hotel_cancel["mean"],
                      color=[PALETTE[2], PALETTE[3]])
        ax.set_title("Cancellation Rate by Hotel Type")
        ax.set_ylabel("Cancellation Rate (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%", ha="center")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Quick Business Summary")
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| Total bookings (cleaned) | {total_bookings:,} |
| Cancelled bookings | {cancelled:,} ({cancel_rate:.1f}%) |
| Not cancelled | {total_bookings - cancelled:,} ({100-cancel_rate:.1f}%) |
| Average ADR | EUR {avg_adr:.2f} |
| Average lead time | {avg_lead_time:.0f} days |
| Average stay duration | {avg_stay:.1f} nights |
| Repeat guest share | {repeat_pct:.1f}% |
    """)

# ── PAGE 2: BOOKING ANALYTICS ─────────────────────────────────────────────────
elif page == "📊 Booking Analytics":
    st.title("📊 Booking Analytics")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Monthly Trends", "Hotel Comparison", "Customer Types", "Market Segments"
    ])

    with tab1:
        st.subheader("Monthly Booking Demand")
        monthly = df.groupby("arrival_date_month").size().reindex(MONTH_ORDER)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(MONTH_ORDER, monthly.values, marker="o", linewidth=2, color=PALETTE[2])
        ax.fill_between(MONTH_ORDER, monthly.values, alpha=0.2, color=PALETTE[2])
        ax.set_title("Monthly Booking Demand")
        ax.set_ylabel("Number of Bookings")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        plt.close()
        st.info(f"**Peak month:** {monthly.idxmax()} ({monthly.max():,} bookings)  |  "
                f"**Lowest month:** {monthly.idxmin()} ({monthly.min():,} bookings)")

        st.subheader("Bookings by Year & Month")
        year_month = df.groupby(["arrival_date_year", "arrival_date_month"]).size().reset_index(name="count")
        year_month["month_num"] = year_month["arrival_date_month"].map({
            m: i+1 for i, m in enumerate(MONTH_ORDER)
        })
        year_month = year_month.sort_values(["arrival_date_year", "month_num"])

        fig, ax = plt.subplots(figsize=(12, 5))
        for yr, grp in year_month.groupby("arrival_date_year"):
            ax.plot(grp["arrival_date_month"], grp["count"], marker="o",
                    linewidth=2, label=str(yr))
        ax.set_title("Bookings by Year and Month")
        ax.set_ylabel("Bookings")
        ax.legend()
        plt.xticks(rotation=30)
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.subheader("Hotel Type Comparison")
        col1, col2 = st.columns(2)
        with col1:
            hotel_counts = df["hotel"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(hotel_counts.index, hotel_counts.values, color=[PALETTE[2], PALETTE[3]])
            ax.set_title("Total Bookings by Hotel Type")
            ax.set_ylabel("Number of Bookings")
            for i, v in enumerate(hotel_counts.values):
                ax.text(i, v + 50, f"{v:,}", ha="center")
            st.pyplot(fig)
            plt.close()

        with col2:
            hotel_adr = df.groupby("hotel")["adr"].mean()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(hotel_adr.index, hotel_adr.values, color=[PALETTE[0], PALETTE[4]])
            ax.set_title("Average Daily Rate by Hotel Type")
            ax.set_ylabel("ADR (EUR)")
            for i, v in enumerate(hotel_adr.values):
                ax.text(i, v + 0.5, f"EUR {v:.1f}", ha="center")
            st.pyplot(fig)
            plt.close()

        st.subheader("Lead Time & Length of Stay Comparison")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, (col_name, label) in enumerate([("lead_time", "Lead Time (days)"),
                                               ("stays_in_week_nights", "Weekday Nights")]):
            for hotel, color in zip(["City Hotel", "Resort Hotel"], [PALETTE[2], PALETTE[3]]):
                axes[i].hist(df[df["hotel"]==hotel][col_name], bins=30,
                             alpha=0.6, label=hotel, color=color)
            axes[i].set_title(f"{label} by Hotel Type")
            axes[i].set_xlabel(label)
            axes[i].set_ylabel("Count")
            axes[i].legend()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.subheader("Customer Type Analysis")
        col1, col2 = st.columns(2)
        with col1:
            cust_counts = df["customer_type"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(cust_counts.values, labels=cust_counts.index,
                   autopct="%1.1f%%", colors=PALETTE, startangle=90)
            ax.set_title("Customer Type Distribution")
            st.pyplot(fig)
            plt.close()
        with col2:
            st.dataframe(
                df.groupby("customer_type").agg(
                    Bookings=("is_canceled", "count"),
                    Cancellation_Rate=("is_canceled", lambda x: f"{x.mean()*100:.1f}%"),
                    Avg_ADR=("adr", lambda x: f"EUR {x.mean():.1f}"),
                    Avg_Lead_Time=("lead_time", lambda x: f"{x.mean():.0f} days")
                ).reset_index(),
                use_container_width=True
            )

        st.subheader("Repeat vs New Guests")
        repeat_stats = df.groupby("is_repeated_guest").agg(
            Count=("is_canceled", "count"),
            Cancel_Rate=("is_canceled", lambda x: f"{x.mean()*100:.1f}%"),
        )
        repeat_stats.index = ["New Guest", "Repeat Guest"]
        st.dataframe(repeat_stats, use_container_width=True)

    with tab4:
        st.subheader("Market Segment Analysis")
        col1, col2 = st.columns(2)
        with col1:
            seg_counts = df["market_segment"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(seg_counts.index, seg_counts.values, color=PALETTE)
            ax.set_title("Bookings by Market Segment")
            ax.set_xlabel("Number of Bookings")
            st.pyplot(fig)
            plt.close()
        with col2:
            seg_adr = df.groupby("market_segment")["adr"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(seg_adr.index, seg_adr.values, color=PALETTE[2])
            ax.set_title("Average ADR by Market Segment")
            ax.set_xlabel("ADR (EUR)")
            st.pyplot(fig)
            plt.close()

        st.subheader("Monthly ADR Trend")
        monthly_adr = df.groupby("arrival_date_month")["adr"].mean().reindex(MONTH_ORDER)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(MONTH_ORDER, monthly_adr.values, marker="o", linewidth=2, color=PALETTE[4])
        ax.fill_between(MONTH_ORDER, monthly_adr.values, alpha=0.2, color=PALETTE[4])
        ax.set_title("Average Daily Rate by Month")
        ax.set_ylabel("ADR (EUR)")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        plt.close()

# ── PAGE 3: CANCELLATION ANALYTICS ───────────────────────────────────────────
elif page == "❌ Cancellation Analytics":
    st.title("❌ Cancellation Analytics")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Cancellation Rate", f"{df['is_canceled'].mean()*100:.1f}%")
    col2.metric("City Hotel Cancel Rate",
                f"{df[df['hotel']=='City Hotel']['is_canceled'].mean()*100:.1f}%")
    col3.metric("Resort Hotel Cancel Rate",
                f"{df[df['hotel']=='Resort Hotel']['is_canceled'].mean()*100:.1f}%")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["By Category", "Lead Time Analysis", "Patterns"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Cancellation by Deposit Type")
            dep_cancel = df.groupby("deposit_type")["is_canceled"].mean().sort_values(ascending=False) * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(dep_cancel.index, dep_cancel.values,
                          color=[PALETTE[1], PALETTE[0], PALETTE[2]])
            ax.set_title("Cancellation Rate by Deposit Type")
            ax.set_ylabel("Cancellation Rate (%)")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{bar.get_height():.1f}%", ha="center")
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.subheader("Cancellation by Customer Type")
            cust_cancel = df.groupby("customer_type")["is_canceled"].mean().sort_values(ascending=False) * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(cust_cancel.index, cust_cancel.values, color=PALETTE)
            ax.set_title("Cancellation Rate by Customer Type")
            ax.set_ylabel("Cancellation Rate (%)")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            plt.xticks(rotation=15)
            st.pyplot(fig)
            plt.close()

        st.subheader("Cancellation Rate by Market Segment")
        seg_cancel = df.groupby("market_segment")["is_canceled"].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(seg_cancel.index, seg_cancel.values, color=PALETTE[1])
        ax.set_title("Cancellation Rate by Market Segment")
        ax.set_xlabel("Cancellation Rate (%)")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        st.pyplot(fig)
        plt.close()

        st.subheader("Monthly Cancellation Rate")
        monthly_cancel = df.groupby("arrival_date_month")["is_canceled"].mean().reindex(MONTH_ORDER) * 100
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(MONTH_ORDER, monthly_cancel.values, color=PALETTE[1], alpha=0.8)
        ax.set_title("Monthly Cancellation Rate")
        ax.set_ylabel("Cancellation Rate (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        plt.xticks(rotation=30)
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.subheader("Lead Time vs Cancellation")
        lt_bins = pd.cut(df["lead_time"], bins=[0, 7, 30, 90, 180, 737],
                         labels=["0-7d", "8-30d", "31-90d", "91-180d", ">180d"])
        lt_cancel = df.groupby(lt_bins, observed=True)["is_canceled"].mean() * 100
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(lt_cancel.index.astype(str), lt_cancel.values, color=PALETTE[2])
        ax.set_title("Cancellation Rate by Lead Time Bucket")
        ax.set_ylabel("Cancellation Rate (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        for i, v in enumerate(lt_cancel.values):
            ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
        st.pyplot(fig)
        plt.close()
        st.info("Bookings made far in advance (>180 days) show the highest cancellation rates, "
                "suggesting that very early commitments are more likely to be cancelled.")

    with tab3:
        st.subheader("Special Requests vs Cancellation")
        req_cancel = df.groupby("total_of_special_requests")["is_canceled"].mean() * 100
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(req_cancel.index.astype(str), req_cancel.values, color=PALETTE[0])
        ax.set_title("Cancellation Rate by Number of Special Requests")
        ax.set_ylabel("Cancellation Rate (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        st.pyplot(fig)
        plt.close()
        st.info("More special requests are associated with lower cancellation probability — "
                "guests with specific needs are more committed to the booking.")

        st.subheader("Previous Cancellation History")
        prev_bins = pd.cut(df["previous_cancellations"], bins=[-1, 0, 1, 5, 27],
                           labels=["None", "1", "2-5", ">5"])
        prev_cancel = df.groupby(prev_bins, observed=True)["is_canceled"].mean() * 100
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(prev_cancel.index.astype(str), prev_cancel.values, color=PALETTE[3])
        ax.set_title("Current Cancellation Rate by Previous Cancellation Count")
        ax.set_ylabel("Cancellation Rate (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        st.pyplot(fig)
        plt.close()
        st.info("Guests with a history of cancellations show a much higher rate "
                "of cancelling their current booking.")

# ── PAGE 4: ML PREDICTION ─────────────────────────────────────────────────────
elif page == "🤖 ML Prediction":
    st.title("🤖 Cancellation Prediction")
    st.markdown("Enter booking details to predict whether it is likely to be cancelled.")
    st.markdown("---")

    try:
        model, feature_columns = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Model could not be loaded: {e}")
        model_loaded = False

    if model_loaded:
        with st.form("prediction_form"):
            st.subheader("Booking Information")

            col1, col2, col3 = st.columns(3)
            with col1:
                hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
                lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=737, value=60)
                arrival_month = st.selectbox("Arrival Month", MONTH_ORDER, index=6)
                stays_weekend = st.number_input("Weekend Nights", min_value=0, max_value=19, value=1)
                stays_week    = st.number_input("Weekday Nights",  min_value=0, max_value=50, value=2)

            with col2:
                adults     = st.number_input("Adults",   min_value=1, max_value=55, value=2)
                children   = st.number_input("Children", min_value=0, max_value=10, value=0)
                babies     = st.number_input("Babies",   min_value=0, max_value=10, value=0)
                meal       = st.selectbox("Meal Plan", ["BB", "HB", "FB", "SC"])
                market_seg = st.selectbox("Market Segment",
                    ["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate",
                     "Complementary", "Aviation", "Undefined"])

            with col3:
                deposit_type = st.selectbox("Deposit Type",
                    ["No Deposit", "Non Refund", "Refundable"])
                customer_type = st.selectbox("Customer Type",
                    ["Transient", "Transient-Party", "Contract", "Group"])
                reserved_room = st.selectbox("Reserved Room Type",
                    ["A", "B", "C", "D", "E", "F", "G", "H", "L"])
                assigned_room = st.selectbox("Assigned Room Type",
                    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L"])
                dist_channel = st.selectbox("Distribution Channel",
                    ["TA/TO", "Direct", "Corporate", "GDS"])
                adr            = st.number_input("ADR (EUR)", min_value=0.0, max_value=5000.0, value=100.0)
                special_req    = st.number_input("Special Requests", min_value=0, max_value=5, value=0)
                prev_cancel    = st.number_input("Previous Cancellations", min_value=0, max_value=26, value=0)
                prev_not_cancel= st.number_input("Previous Bookings Not Cancelled", min_value=0, max_value=72, value=0)
                parking        = st.number_input("Parking Spaces Required", min_value=0, max_value=8, value=0)
                booking_chg    = st.number_input("Booking Changes", min_value=0, max_value=21, value=0)
                wait_list      = st.number_input("Days in Waiting List", min_value=0, max_value=391, value=0)
                is_repeated    = st.selectbox("Is Repeated Guest", [0, 1], format_func=lambda x: "Yes" if x else "No")

            submitted = st.form_submit_button("🔍 Predict Cancellation", use_container_width=True)

        if submitted:
            month_map = {m: i+1 for i, m in enumerate(MONTH_ORDER)}
            arrival_month_num = month_map[arrival_month]
            total_stay_nights = stays_weekend + stays_week
            total_guests      = adults + children + babies
            is_family         = int(children > 0 or babies > 0)
            is_weekend_bk     = int(stays_weekend > 0)
            has_agent         = 0  # form doesn't collect agent ID
            has_company       = 0  # form doesn't collect company ID
            total_prev        = prev_cancel + prev_not_cancel
            cancel_hist_rate  = prev_cancel / total_prev if total_prev > 0 else 0.0
            room_type_match   = int(reserved_room == assigned_room)

            # lead_time_bin
            if lead_time <= 7:
                lt_bin = "same_week"
            elif lead_time <= 30:
                lt_bin = "short"
            elif lead_time <= 90:
                lt_bin = "medium"
            elif lead_time <= 180:
                lt_bin = "long"
            else:
                lt_bin = "very_long"

            adr_per_night = adr if total_stay_nights > 0 else 0.0

            input_dict = {
                "lead_time": lead_time,
                "stays_in_weekend_nights": stays_weekend,
                "stays_in_week_nights": stays_week,
                "adults": adults,
                "children": children,
                "babies": babies,
                "is_repeated_guest": is_repeated,
                "previous_cancellations": prev_cancel,
                "previous_bookings_not_canceled": prev_not_cancel,
                "booking_changes": booking_chg,
                "days_in_waiting_list": wait_list,
                "adr": adr,
                "required_car_parking_spaces": parking,
                "total_of_special_requests": special_req,
                "total_stay_nights": total_stay_nights,
                "total_guests": total_guests,
                "is_family": is_family,
                "is_weekend_booking": is_weekend_bk,
                "arrival_month_num": arrival_month_num,
                "has_company": has_company,
                "has_agent": has_agent,
                "cancellation_history_rate": cancel_hist_rate,
                "room_type_match": room_type_match,
                "adr_per_night": adr_per_night,
                "hotel": hotel,
                "meal": meal,
                "market_segment": market_seg,
                "distribution_channel": dist_channel,
                "reserved_room_type": reserved_room,
                "assigned_room_type": assigned_room,
                "deposit_type": deposit_type,
                "customer_type": customer_type,
                "lead_time_bin": lt_bin,
            }

            input_df = pd.DataFrame([input_dict])
            # Align columns to what the model expects
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_columns]

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] * 100

            st.markdown("---")
            st.subheader("Prediction Result")

            if prediction == 1:
                st.error(f"### ❌ Likely to be CANCELLED\n**Cancellation Probability: {probability:.1f}%**")
                st.markdown("""
**Key risk factors that may have influenced this prediction:**
- High lead time (bookings made far in advance have higher cancellation risk)
- Non-refundable deposit (counter-intuitively, non-refundable bookings show high cancellation)
- History of previous cancellations
- Low special request count
                """)
            else:
                st.success(f"### ✅ Likely to be HONOURED\n**Cancellation Probability: {probability:.1f}%**")
                st.markdown("""
**Factors reducing cancellation risk:**
- Short lead time (booking closer to arrival date)
- No prior cancellation history
- Special requests submitted (indicates commitment)
- Repeat guest (loyal customers rarely cancel)
                """)

            # Probability gauge
            fig, ax = plt.subplots(figsize=(7, 2))
            ax.barh(["Cancellation\nProbability"], [probability], color=PALETTE[1] if prediction == 1 else PALETTE[0])
            ax.barh(["Cancellation\nProbability"], [100 - probability], left=[probability],
                    color="#e0e0e0", alpha=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Cancellation Probability")
            ax.text(probability/2, 0, f"{probability:.1f}%", ha="center", va="center",
                    fontweight="bold", color="white", fontsize=12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.caption("""
**Model Information:**  
Best model: Gradient Boosting | ROC-AUC = 0.8623 | Accuracy = 0.8178  
*Predictions are based on historical patterns and should be used as a decision-support tool, not as definitive outcomes.*
        """)
