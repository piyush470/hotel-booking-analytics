"""
generate_dashboard_screenshots.py
----------------------------------
Renders static mockup screenshots of all 4 dashboard pages
and saves them to images/dashboard_*.png for embedding in the Word report.
Run from project root: python generate_dashboard_screenshots.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from src.data_cleaning import load_raw_data, clean_data
from src.feature_engineering import engineer_features

# ── Load data ─────────────────────────────────────────────────
df = pd.read_csv("data/processed/cleaned_hotel_bookings.csv")
PALETTE = ["#4CAF50", "#F44336", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4", "#E91E63", "#8BC34A"]
MONTH_ORDER = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
IMG = Path("images")

# ─────────────────────────────────────────────────────────────
# PAGE 1 — Executive Overview
# ─────────────────────────────────────────────────────────────
def page1():
    total = len(df)
    cancelled = int(df["is_canceled"].sum())
    cancel_rate = df["is_canceled"].mean() * 100
    avg_adr = df["adr"].mean()
    avg_lead = df["lead_time"].mean()
    avg_stay = (df["stays_in_weekend_nights"] + df["stays_in_week_nights"]).mean()
    city_pct = (df["hotel"] == "City Hotel").mean() * 100
    repeat_pct = df["is_repeated_guest"].mean() * 100

    fig = plt.figure(figsize=(16, 10), facecolor="#0e1117")
    fig.suptitle("🏨  Hotel Booking Analytics — Executive Overview",
                 color="white", fontsize=15, fontweight="bold", y=0.97)

    # KPI cards (top two rows)
    kpis = [
        ("Total Bookings", f"{total:,}", "#2196F3"),
        ("Cancellation Rate", f"{cancel_rate:.1f}%", "#F44336"),
        ("Cancelled Bookings", f"{cancelled:,}", "#FF9800"),
        ("Average ADR (EUR)", f"{avg_adr:.2f}", "#4CAF50"),
        ("Avg Lead Time (days)", f"{avg_lead:.0f}", "#9C27B0"),
        ("Avg Stay (nights)", f"{avg_stay:.1f}", "#00BCD4"),
        ("City Hotel Share", f"{city_pct:.1f}%", "#2196F3"),
        ("Repeat Guests", f"{repeat_pct:.1f}%", "#8BC34A"),
    ]
    for i, (label, value, color) in enumerate(kpis):
        ax = fig.add_axes([0.03 + (i % 4) * 0.245, 0.75 - (i // 4) * 0.13, 0.22, 0.10])
        ax.set_facecolor("#1e2130")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch((0.02, 0.05), 0.96, 0.90,
            boxstyle="round,pad=0.02", linewidth=2,
            edgecolor=color, facecolor="#1e2130"))
        ax.text(0.5, 0.72, value, ha="center", va="center",
                color=color, fontsize=16, fontweight="bold")
        ax.text(0.5, 0.28, label, ha="center", va="center",
                color="#aaaaaa", fontsize=8)

    # Pie chart
    ax_pie = fig.add_axes([0.04, 0.08, 0.28, 0.50])
    ax_pie.set_facecolor("#1e2130")
    counts = df["is_canceled"].value_counts()
    ax_pie.pie(counts.values, labels=["Not Cancelled", "Cancelled"],
               autopct="%1.1f%%", colors=[PALETTE[0], PALETTE[1]],
               startangle=90, textprops={"color": "white", "fontsize": 9})
    ax_pie.set_title("Cancellation Distribution", color="white", fontsize=10, pad=8)

    # Bar: cancel rate by hotel
    ax_bar = fig.add_axes([0.37, 0.08, 0.28, 0.50])
    ax_bar.set_facecolor("#1e2130")
    hotel_cr = df.groupby("hotel")["is_canceled"].mean() * 100
    bars = ax_bar.bar(hotel_cr.index, hotel_cr.values, color=[PALETTE[2], PALETTE[3]], width=0.5)
    ax_bar.set_facecolor("#1e2130")
    ax_bar.set_title("Cancellation Rate by Hotel", color="white", fontsize=10)
    ax_bar.set_ylabel("Cancellation Rate (%)", color="#aaaaaa", fontsize=8)
    ax_bar.tick_params(colors="white", labelsize=8)
    ax_bar.spines[:].set_color("#333333")
    ax_bar.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    for bar in bars:
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%", ha="center", color="white", fontsize=9)

    # Summary table
    ax_tbl = fig.add_axes([0.68, 0.08, 0.30, 0.50])
    ax_tbl.set_facecolor("#1e2130")
    ax_tbl.axis("off")
    rows = [
        ["Total bookings", f"{total:,}"],
        ["Cancelled", f"{cancelled:,} ({cancel_rate:.1f}%)"],
        ["Not cancelled", f"{total-cancelled:,} ({100-cancel_rate:.1f}%)"],
        ["Average ADR", f"EUR {avg_adr:.2f}"],
        ["Avg lead time", f"{avg_lead:.0f} days"],
        ["Avg stay", f"{avg_stay:.1f} nights"],
        ["Repeat guests", f"{repeat_pct:.1f}%"],
    ]
    tbl = ax_tbl.table(cellText=rows, colLabels=["Metric", "Value"],
                       cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#262b3d" if r % 2 == 0 else "#1e2130")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333333")
        if r == 0:
            cell.set_facecolor("#2196F3")
            cell.set_text_props(color="white", fontweight="bold")
    ax_tbl.set_title("Quick Summary", color="white", fontsize=10, pad=8)

    plt.savefig(IMG / "dashboard_page1_overview.png", bbox_inches="tight",
                facecolor="#0e1117", dpi=120)
    plt.close()
    print("  Saved dashboard_page1_overview.png")


# ─────────────────────────────────────────────────────────────
# PAGE 2 — Booking Analytics
# ─────────────────────────────────────────────────────────────
def page2():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#0e1117")
    fig.suptitle("📊  Booking Analytics", color="white", fontsize=15,
                 fontweight="bold", y=0.97)

    monthly = df.groupby("arrival_date_month").size().reindex(MONTH_ORDER)
    hotel_counts = df["hotel"].value_counts()
    seg_counts = df["market_segment"].value_counts()
    monthly_adr = df.groupby("arrival_date_month")["adr"].mean().reindex(MONTH_ORDER)

    plots = [
        (axes[0,0], "Monthly Booking Demand", "line"),
        (axes[0,1], "Bookings by Hotel Type", "bar_hotel"),
        (axes[1,0], "Bookings by Market Segment", "barh_seg"),
        (axes[1,1], "Average Daily Rate by Month", "line_adr"),
    ]

    for ax, title, kind in plots:
        ax.set_facecolor("#1e2130")
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[:].set_color("#333333")
        for spine in ax.spines.values():
            spine.set_color("#333333")

        if kind == "line":
            ax.plot(range(12), monthly.values, marker="o", color=PALETTE[2], linewidth=2)
            ax.fill_between(range(12), monthly.values, alpha=0.2, color=PALETTE[2])
            ax.set_xticks(range(12))
            ax.set_xticklabels([m[:3] for m in MONTH_ORDER], rotation=30, color="white")
            ax.set_ylabel("Bookings", color="#aaaaaa", fontsize=8)
            ax.yaxis.label.set_color("#aaaaaa")
        elif kind == "bar_hotel":
            bars = ax.bar(hotel_counts.index, hotel_counts.values,
                          color=[PALETTE[2], PALETTE[3]], width=0.5)
            ax.set_ylabel("Bookings", color="#aaaaaa", fontsize=8)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                        f"{bar.get_height():,}", ha="center", color="white", fontsize=9)
        elif kind == "barh_seg":
            ax.barh(seg_counts.index, seg_counts.values, color=PALETTE)
            ax.set_xlabel("Bookings", color="#aaaaaa", fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
        elif kind == "line_adr":
            ax.plot(range(12), monthly_adr.values, marker="o", color=PALETTE[4], linewidth=2)
            ax.fill_between(range(12), monthly_adr.values, alpha=0.2, color=PALETTE[4])
            ax.set_xticks(range(12))
            ax.set_xticklabels([m[:3] for m in MONTH_ORDER], rotation=30, color="white")
            ax.set_ylabel("ADR (EUR)", color="#aaaaaa", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(IMG / "dashboard_page2_booking.png", bbox_inches="tight",
                facecolor="#0e1117", dpi=120)
    plt.close()
    print("  Saved dashboard_page2_booking.png")


# ─────────────────────────────────────────────────────────────
# PAGE 3 — Cancellation Analytics
# ─────────────────────────────────────────────────────────────
def page3():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#0e1117")
    fig.suptitle("❌  Cancellation Analytics", color="white", fontsize=15,
                 fontweight="bold", y=0.97)

    dep_cr = df.groupby("deposit_type")["is_canceled"].mean().sort_values(ascending=False) * 100
    cust_cr = df.groupby("customer_type")["is_canceled"].mean().sort_values(ascending=False) * 100
    lt_bins = pd.cut(df["lead_time"], bins=[0,7,30,90,180,737],
                     labels=["0-7d","8-30d","31-90d","91-180d",">180d"])
    lt_cr = df.groupby(lt_bins, observed=True)["is_canceled"].mean() * 100
    monthly_cr = df.groupby("arrival_date_month")["is_canceled"].mean().reindex(MONTH_ORDER) * 100

    datasets = [
        (axes[0,0], "Cancel Rate by Deposit Type", dep_cr, "bar"),
        (axes[0,1], "Cancel Rate by Customer Type", cust_cr, "bar"),
        (axes[1,0], "Cancel Rate by Lead Time Bucket", lt_cr, "bar"),
        (axes[1,1], "Monthly Cancellation Rate", monthly_cr, "bar_monthly"),
    ]

    for ax, title, data, kind in datasets:
        ax.set_facecolor("#1e2130")
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333333")

        if kind == "bar":
            bars = ax.bar(data.index, data.values, color=PALETTE[1], alpha=0.85)
            ax.set_ylabel("Cancellation Rate (%)", color="#aaaaaa", fontsize=8)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{bar.get_height():.1f}%", ha="center", color="white", fontsize=8)
            ax.tick_params(axis="x", labelsize=7, rotation=10)
        elif kind == "bar_monthly":
            ax.bar(range(12), data.values, color=PALETTE[1], alpha=0.85)
            ax.set_xticks(range(12))
            ax.set_xticklabels([m[:3] for m in MONTH_ORDER], rotation=30, color="white")
            ax.set_ylabel("Cancellation Rate (%)", color="#aaaaaa", fontsize=8)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(IMG / "dashboard_page3_cancellation.png", bbox_inches="tight",
                facecolor="#0e1117", dpi=120)
    plt.close()
    print("  Saved dashboard_page3_cancellation.png")


# ─────────────────────────────────────────────────────────────
# PAGE 4 — ML Prediction Form
# ─────────────────────────────────────────────────────────────
def page4():
    fig = plt.figure(figsize=(16, 10), facecolor="#0e1117")
    fig.suptitle("🤖  Cancellation Prediction — ML Model",
                 color="white", fontsize=15, fontweight="bold", y=0.97)

    # Form panel (left)
    ax_form = fig.add_axes([0.02, 0.06, 0.44, 0.88])
    ax_form.set_facecolor("#1e2130")
    ax_form.axis("off")
    ax_form.add_patch(mpatches.FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.01", linewidth=1.5,
        edgecolor="#3a7bd5", facecolor="#1e2130"))
    ax_form.text(0.5, 0.95, "Booking Information", ha="center", va="top",
                 color="white", fontsize=12, fontweight="bold")

    fields = [
        ("Hotel Type",               "City Hotel"),
        ("Lead Time (days)",         "120"),
        ("Arrival Month",            "August"),
        ("Weekend Nights",           "2"),
        ("Weekday Nights",           "3"),
        ("Adults",                   "2"),
        ("Children",                 "0"),
        ("Meal Plan",                "BB"),
        ("Market Segment",           "Online TA"),
        ("Deposit Type",             "No Deposit"),
        ("Customer Type",            "Transient"),
        ("Reserved Room Type",       "A"),
        ("ADR (EUR)",                "95.00"),
        ("Special Requests",         "1"),
        ("Previous Cancellations",   "0"),
    ]
    y_start = 0.88
    for label, val in fields:
        ax_form.text(0.05, y_start, label + ":", color="#aaaaaa", fontsize=8, va="top")
        ax_form.add_patch(mpatches.FancyBboxPatch(
            (0.45, y_start - 0.025), 0.50, 0.035,
            boxstyle="round,pad=0.005", linewidth=0.8,
            edgecolor="#3a7bd5", facecolor="#262b3d"))
        ax_form.text(0.70, y_start - 0.007, val, color="white", fontsize=8, va="top", ha="center")
        y_start -= 0.055

    ax_form.add_patch(mpatches.FancyBboxPatch((0.15, 0.03), 0.70, 0.055,
        boxstyle="round,pad=0.01", linewidth=1,
        edgecolor="#3a7bd5", facecolor="#3a7bd5"))
    ax_form.text(0.5, 0.057, "🔍  Predict Cancellation", ha="center", va="center",
                 color="white", fontsize=10, fontweight="bold")

    # Result panel (right)
    ax_res = fig.add_axes([0.50, 0.06, 0.48, 0.88])
    ax_res.set_facecolor("#1e2130")
    ax_res.axis("off")
    ax_res.add_patch(mpatches.FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.01", linewidth=1.5,
        edgecolor="#4CAF50", facecolor="#1e2130"))
    ax_res.text(0.5, 0.95, "Prediction Result", ha="center", va="top",
                color="white", fontsize=12, fontweight="bold")

    ax_res.add_patch(mpatches.FancyBboxPatch((0.05, 0.72), 0.90, 0.15,
        boxstyle="round,pad=0.01", linewidth=2,
        edgecolor="#4CAF50", facecolor="#1a3a1a"))
    ax_res.text(0.5, 0.82, "✅  Likely to be HONOURED",
                ha="center", va="center", color="#4CAF50", fontsize=13, fontweight="bold")
    ax_res.text(0.5, 0.74, "Cancellation Probability:  23.2%",
                ha="center", va="center", color="white", fontsize=10)

    # Probability bar
    ax_res.text(0.5, 0.66, "Cancellation Probability", ha="center",
                color="#aaaaaa", fontsize=9)
    ax_res.add_patch(mpatches.FancyBboxPatch((0.05, 0.59), 0.90, 0.055,
        boxstyle="round,pad=0.003", linewidth=0,
        edgecolor="none", facecolor="#333333"))
    ax_res.add_patch(mpatches.FancyBboxPatch((0.05, 0.59), 0.90 * 0.232, 0.055,
        boxstyle="round,pad=0.003", linewidth=0,
        edgecolor="none", facecolor="#4CAF50"))
    ax_res.text(0.50 * 0.232 + 0.05, 0.617, "23.2%", ha="center",
                color="white", fontsize=8, fontweight="bold")

    # Factors
    ax_res.text(0.5, 0.53, "Factors Reducing Cancellation Risk", ha="center",
                color="white", fontsize=9, fontweight="bold")
    reasons = [
        "• Short lead time (closer to arrival date)",
        "• No prior cancellation history",
        "• Special request submitted",
        "• No Deposit — lower financial commitment",
    ]
    for i, r in enumerate(reasons):
        ax_res.text(0.08, 0.48 - i * 0.055, r, color="#aaaaaa", fontsize=8)

    ax_res.text(0.5, 0.10,
                "Model: Gradient Boosting  |  ROC-AUC = 0.8623  |  Accuracy = 81.78%",
                ha="center", color="#555555", fontsize=7.5, style="italic")

    plt.savefig(IMG / "dashboard_page4_prediction.png", bbox_inches="tight",
                facecolor="#0e1117", dpi=120)
    plt.close()
    print("  Saved dashboard_page4_prediction.png")


if __name__ == "__main__":
    print("Generating dashboard UI screenshots...")
    page1()
    page2()
    page3()
    page4()
    print("Done — 4 screenshots saved to images/")
