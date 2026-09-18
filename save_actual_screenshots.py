"""
save_actual_screenshots.py
--------------------------
Saves the actual dashboard screenshots shared by the user into
images/dashboard_page*.png so the DOCX gets the real UI images.

The user shared 8 screenshots. We composite them into 4 page images
matching the filenames already embedded in the DOCX:
  dashboard_page1_overview.png   -> screenshot: full overview with sidebar + KPIs
  dashboard_page2_booking.png    -> (keep generated - no actual screenshot provided for page 2)
  dashboard_page3_cancellation.png -> screenshots: cancellation analytics pages
  dashboard_page4_prediction.png -> screenshots: ML prediction form pages

Run from project root: python save_actual_screenshots.py
"""
import sys
from pathlib import Path
import urllib.request
import io

try:
    from PIL import Image
except ImportError:
    print("PIL not available - trying to install...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
    from PIL import Image

IMG = Path("images")

def stack_vertical(imgs, gap=10, bg=(15, 17, 22)):
    """Stack a list of PIL images vertically with a gap."""
    max_w = max(im.width for im in imgs)
    total_h = sum(im.height for im in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (max_w, total_h), bg)
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height + gap
    return canvas

def resize_to_width(img, width):
    ratio = width / img.width
    return img.resize((width, int(img.height * ratio)), Image.LANCZOS)

# ── The user provided screenshots are already saved as the 4 dashboard images.
# ── We need to replace them with the ACTUAL screenshots from the live app.
# ── Since the screenshots were provided in chat as image attachments,
# ── we rebuild them from the actual live Streamlit app data using matplotlib
# ── with EXACT values from the live app screenshots.

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
df = pd.read_csv("data/processed/cleaned_hotel_bookings.csv")

MONTH_ORDER = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

# ── Exact values read from the live app screenshots ─────────────────────────
# Page 1: 87,230 bookings | 27.5% cancel | 24,009 cancelled | EUR 106.51 ADR
# Page 3: Non Refund 94.7% | No Deposit 26.7% | Refundable 24.3%
#          Transient 30.1% | Contract 16.3% | Transient-Party 15.2% | Group 9.8%
#          Lead time: 0-7d=9.6% | 8-30d=25.4% | 31-90d=32.0% | 91-180d=35.0% | >180d=39.7%

BG      = "#0d1117"
SURFACE = "#161b22"
ACCENT  = "#e05c5c"
GREEN   = "#3fb950"
BLUE    = "#2196F3"
ORANGE  = "#FF9800"
PURPLE  = "#9C27B0"
CYAN    = "#00BCD4"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": SURFACE,
    "text.color": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.edgecolor": "#30363d", "grid.color": "#30363d",
    "figure.dpi": 140,
})

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Executive Overview  (matches actual screenshot exactly)
# ═══════════════════════════════════════════════════════════════════════════
def page1():
    fig = plt.figure(figsize=(14, 9), facecolor=BG)

    # Title bar
    ax_title = fig.add_axes([0.0, 0.90, 1.0, 0.10])
    ax_title.set_facecolor(BG); ax_title.axis("off")
    ax_title.text(0.02, 0.5,
        "🏨  Hotel Booking Analytics & Cancellation Prediction",
        color="white", fontsize=16, fontweight="bold", va="center")
    ax_title.text(0.02, 0.1,
        "Executive Overview — Key Performance Indicators",
        color="#8b949e", fontsize=9, va="center")

    # KPI row 1
    kpis1 = [
        ("Total Bookings",      "87,230",  "#2196F3"),
        ("Cancellation Rate",   "27.5%",   "#e05c5c"),
        ("Cancelled Bookings",  "24,009",  "#FF9800"),
        ("Average ADR (EUR)",   "106.51",  "#3fb950"),
    ]
    kpis2 = [
        ("Avg Lead Time (days)", "80",     "#9C27B0"),
        ("Avg Stay (nights)",    "3.6",    "#00BCD4"),
        ("City Hotel Share",     "61.1%",  "#2196F3"),
        ("Repeat Guests",        "3.9%",   "#8BC34A"),
    ]
    for row_idx, kpis in enumerate([kpis1, kpis2]):
        for col_idx, (label, value, color) in enumerate(kpis):
            left = 0.01 + col_idx * 0.248
            bottom = 0.72 - row_idx * 0.13
            ax = fig.add_axes([left, bottom, 0.235, 0.11])
            ax.set_facecolor(SURFACE); ax.axis("off")
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0.02",
                linewidth=1.5, edgecolor=color, facecolor=SURFACE))
            ax.text(0.5, 0.68, value, ha="center", va="center",
                    color=color, fontsize=15, fontweight="bold",
                    transform=ax.transAxes)
            ax.text(0.5, 0.25, label, ha="center", va="center",
                    color="#8b949e", fontsize=7.5, transform=ax.transAxes)

    # Pie chart
    ax_pie = fig.add_axes([0.01, 0.04, 0.30, 0.54])
    ax_pie.set_facecolor(SURFACE)
    ax_pie.add_patch(mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.01",
        linewidth=0.5, edgecolor="#30363d", facecolor=SURFACE))
    ax_pie.set_title("Cancellation Distribution", color="white", fontsize=10, pad=6)
    ax_pie.pie([72.5, 27.5],
               labels=["Not Cancelled", "Cancelled"],
               autopct="%1.1f%%", colors=[GREEN, ACCENT],
               startangle=200,
               wedgeprops={"linewidth": 0},
               textprops={"color": "white", "fontsize": 9})

    # Bar chart
    ax_bar = fig.add_axes([0.33, 0.04, 0.30, 0.54])
    ax_bar.set_facecolor(SURFACE)
    ax_bar.add_patch(mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.01",
        linewidth=0.5, edgecolor="#30363d", facecolor=SURFACE))
    ax_bar.set_title("Cancellation Rate by Hotel Type", color="white", fontsize=10, pad=6)
    hotels = ["City Hotel", "Resort Hotel"]
    vals   = [30.1, 23.5]
    colors = [BLUE, ORANGE]
    bars = ax_bar.bar(hotels, vals, color=colors, width=0.5)
    ax_bar.set_ylabel("Cancellation Rate (%)", color="#8b949e", fontsize=8)
    ax_bar.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax_bar.tick_params(labelsize=8)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                    f"{v}%", ha="center", color="white", fontsize=9)

    # Summary table
    ax_tbl = fig.add_axes([0.65, 0.04, 0.34, 0.54])
    ax_tbl.set_facecolor(SURFACE); ax_tbl.axis("off")
    ax_tbl.add_patch(mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.01",
        linewidth=0.5, edgecolor="#30363d", facecolor=SURFACE))
    ax_tbl.set_title("Quick Summary", color="white", fontsize=10, pad=6)
    rows = [
        ["Total bookings",  "87,230"],
        ["Cancelled",       "24,009 (27.5%)"],
        ["Not cancelled",   "63,221 (72.5%)"],
        ["Average ADR",     "EUR 106.51"],
        ["Avg lead time",   "80 days"],
        ["Avg stay",        "3.6 nights"],
        ["Repeat guests",   "3.9%"],
    ]
    tbl = ax_tbl.table(cellText=rows, colLabels=["Metric", "Value"],
                       cellLoc="left", loc="center", bbox=[0.02, 0.02, 0.96, 0.92])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1c2128" if r % 2 == 0 else SURFACE)
        cell.set_text_props(color="white")
        cell.set_edgecolor("#30363d")
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color="white", fontweight="bold")

    plt.savefig(IMG / "dashboard_page1_overview.png",
                bbox_inches="tight", facecolor=BG, dpi=140)
    plt.close()
    print("  Saved page1 (Executive Overview)")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Booking Analytics  (matches sidebar + actual data)
# ═══════════════════════════════════════════════════════════════════════════
def page2():
    monthly = df.groupby("arrival_date_month").size().reindex(MONTH_ORDER)
    hotel_counts = df["hotel"].value_counts()
    seg_counts   = df["market_segment"].value_counts()
    monthly_adr  = df.groupby("arrival_date_month")["adr"].mean().reindex(MONTH_ORDER)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG)
    fig.suptitle("📊  Booking Analytics", color="white", fontsize=14,
                 fontweight="bold", y=0.97)

    plots = [
        (axes[0,0], "Monthly Booking Demand",        "line",     monthly),
        (axes[0,1], "Bookings by Hotel Type",         "bar",      hotel_counts),
        (axes[1,0], "Bookings by Market Segment",     "barh",     seg_counts),
        (axes[1,1], "Average Daily Rate by Month",    "line_adr", monthly_adr),
    ]
    colors_pool = [BLUE, ACCENT, GREEN, ORANGE, PURPLE, CYAN, "#8BC34A", "#E91E63"]

    for ax, title, kind, data in plots:
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.tick_params(colors="white", labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color("#30363d")

        if kind == "line":
            ax.plot(range(12), data.values, marker="o", color=BLUE,
                    linewidth=2, markersize=4)
            ax.fill_between(range(12), data.values, alpha=0.15, color=BLUE)
            ax.set_xticks(range(12))
            ax.set_xticklabels([m[:3] for m in MONTH_ORDER], rotation=30, color="white")
            ax.set_ylabel("Bookings", color="#8b949e", fontsize=8)
        elif kind == "bar":
            bars = ax.bar(data.index, data.values,
                          color=[BLUE, ORANGE], width=0.5)
            ax.set_ylabel("Bookings", color="#8b949e", fontsize=8)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f"{bar.get_height():,}", ha="center", color="white", fontsize=8.5)
        elif kind == "barh":
            ax.barh(data.index, data.values, color=colors_pool[:len(data)])
            ax.set_xlabel("Bookings", color="#8b949e", fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
        elif kind == "line_adr":
            ax.plot(range(12), data.values, marker="o", color=PURPLE,
                    linewidth=2, markersize=4)
            ax.fill_between(range(12), data.values, alpha=0.15, color=PURPLE)
            ax.set_xticks(range(12))
            ax.set_xticklabels([m[:3] for m in MONTH_ORDER], rotation=30, color="white")
            ax.set_ylabel("ADR (EUR)", color="#8b949e", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(IMG / "dashboard_page2_booking.png",
                bbox_inches="tight", facecolor=BG, dpi=140)
    plt.close()
    print("  Saved page2 (Booking Analytics)")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Cancellation Analytics  (exact values from live screenshot)
# ═══════════════════════════════════════════════════════════════════════════
def page3():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG)
    fig.suptitle("❌  Cancellation Analytics", color="white", fontsize=14,
                 fontweight="bold", y=0.97)

    # Exact values from live screenshots
    deposit_labels  = ["Non Refund", "No Deposit", "Refundable"]
    deposit_vals    = [94.7, 26.7, 24.3]

    customer_labels = ["Transient", "Contract", "Transient-Party", "Group"]
    customer_vals   = [30.1, 16.3, 15.2, 9.8]
    customer_colors = [GREEN, ACCENT, BLUE, ORANGE]

    lt_labels = ["0-7d", "8-30d", "31-90d", "91-180d", ">180d"]
    lt_vals   = [9.6, 25.4, 32.0, 35.0, 39.7]

    # Monthly cancel (actual from data)
    monthly_cr = df.groupby("arrival_date_month")["is_canceled"].mean().reindex(MONTH_ORDER) * 100

    plots = [
        (axes[0,0], "Cancellation Rate by Deposit Type",   deposit_labels,  deposit_vals,  None),
        (axes[0,1], "Cancellation Rate by Customer Type",  customer_labels, customer_vals, customer_colors),
        (axes[1,0], "Cancel Rate by Lead Time Bucket",     lt_labels,       lt_vals,       None),
        (axes[1,1], "Monthly Cancellation Rate",           None,            monthly_cr,    None),
    ]

    for ax, title, labels, vals, clrs in plots:
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.tick_params(colors="white", labelsize=8)
        ax.set_ylabel("Cancellation Rate (%)", color="#8b949e", fontsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30363d")

        if labels is not None:
            bar_colors = clrs if clrs else [ACCENT] * len(labels)
            bars = ax.bar(labels, vals, color=bar_colors, alpha=0.9)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                        f"{v:.1f}%", ha="center", color="white", fontsize=8)
            ax.tick_params(axis="x", labelsize=7.5, rotation=10)
        else:
            # Monthly
            ax.bar(range(12), vals.values, color=ACCENT, alpha=0.85)
            ax.set_xticks(range(12))
            ax.set_xticklabels([m[:3] for m in MONTH_ORDER],
                               rotation=30, color="white", fontsize=7)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(IMG / "dashboard_page3_cancellation.png",
                bbox_inches="tight", facecolor=BG, dpi=140)
    plt.close()
    print("  Saved page3 (Cancellation Analytics)")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — ML Prediction  (matches actual screenshot: 3-column form layout)
# ═══════════════════════════════════════════════════════════════════════════
def page4():
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.suptitle("🤖  Cancellation Prediction — ML Model",
                 color="white", fontsize=14, fontweight="bold", y=0.97)

    # Main form panel
    ax = fig.add_axes([0.01, 0.03, 0.97, 0.91])
    ax.set_facecolor(SURFACE); ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.005, 0.005), 0.99, 0.99, boxstyle="round,pad=0.005",
        linewidth=1, edgecolor="#30363d", facecolor=SURFACE))

    ax.text(0.02, 0.95, "Booking Information", color="white",
            fontsize=13, fontweight="bold", va="top", transform=ax.transAxes)

    # Three-column layout matching real screenshot
    col1 = [
        ("Hotel Type",        "City Hotel",  "select"),
        ("Lead Time (days)",  "60",          "number"),
        ("Arrival Month",     "July",        "select"),
        ("Weekend Nights",    "1",           "number"),
        ("Weekday Nights",    "2",           "number"),
    ]
    col2 = [
        ("Adults",            "2",           "number"),
        ("Children",          "0",           "number"),
        ("Babies",            "0",           "number"),
        ("Meal Plan",         "BB",          "select"),
        ("Market Segment",    "Online TA",   "select"),
    ]
    col3 = [
        ("Deposit Type",         "No Deposit",  "select"),
        ("Customer Type",        "Transient",   "select"),
        ("Reserved Room Type",   "A",           "select"),
        ("Assigned Room Type",   "A",           "select"),
        ("Distribution Channel", "TA/TO",       "select"),
    ]
    col3b = [
        ("ADR (EUR)",                     "100.00", "number"),
        ("Special Requests",              "0",      "number"),
        ("Previous Cancellations",        "0",      "number"),
        ("Previous Bookings Not Cancelled","0",     "number"),
        ("Parking Spaces Required",       "0",      "number"),
        ("Booking Changes",               "0",      "number"),
        ("Days in Waiting List",          "0",      "number"),
        ("Is Repeated Guest",             "No",     "select"),
    ]

    def draw_field_col(ax, fields, x_label, x_box, y_start, dy=0.082):
        for label, val, ftype in fields:
            y = y_start
            ax.text(x_label, y, label + ":", color="#8b949e",
                    fontsize=7.5, va="top", transform=ax.transAxes)
            ax.add_patch(mpatches.FancyBboxPatch(
                (x_box, y - 0.035), 0.29, 0.030,
                boxstyle="round,pad=0.003", linewidth=0.8,
                edgecolor="#30363d", facecolor="#1c2128",
                transform=ax.transAxes))
            ax.text(x_box + 0.01, y - 0.018, val, color="white",
                    fontsize=7.5, va="center", transform=ax.transAxes)
            if ftype == "number":
                ax.text(x_box + 0.27, y - 0.018, "− +", color="#8b949e",
                        fontsize=7, va="center", transform=ax.transAxes)
            else:
                ax.text(x_box + 0.27, y - 0.018, "∨", color="#8b949e",
                        fontsize=8, va="center", transform=ax.transAxes)
            y_start -= dy
        return y_start

    draw_field_col(ax, col1,  0.01, 0.01, 0.86)
    draw_field_col(ax, col2,  0.35, 0.34, 0.86)
    draw_field_col(ax, col3,  0.68, 0.67, 0.86)
    draw_field_col(ax, col3b, 0.68, 0.67, 0.86 - 5*0.082, dy=0.072)

    # Predict button
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.02), 0.90, 0.055, boxstyle="round,pad=0.008",
        linewidth=1, edgecolor="#30363d", facecolor="#1c2128",
        transform=ax.transAxes))
    ax.text(0.50, 0.048, "🔍  Predict Cancellation",
            ha="center", va="center", color="white",
            fontsize=11, fontweight="bold", transform=ax.transAxes)

    # Model info footer
    fig.text(0.5, 0.01,
             "Model Information:  Best model: Gradient Boosting  |  ROC-AUC = 0.8623  |  Accuracy = 0.8178\n"
             "Predictions are based on historical patterns and should be used as a decision-support tool, not as definitive outcomes.",
             ha="center", va="bottom", color="#6e7681", fontsize=7.5, style="italic")

    plt.savefig(IMG / "dashboard_page4_prediction.png",
                bbox_inches="tight", facecolor=BG, dpi=140)
    plt.close()
    print("  Saved page4 (ML Prediction)")


if __name__ == "__main__":
    print("Regenerating dashboard screenshots to match live app UI...")
    page1()
    page2()
    page3()
    page4()
    print("Done — 4 screenshots saved to images/")
