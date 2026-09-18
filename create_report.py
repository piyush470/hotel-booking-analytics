"""
create_report.py
----------------
Generates: Piyush_Hotel_Booking_Project_Report.docx
Run from project root: python create_report.py
"""

from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
import docx.oxml as oxml


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    return p


def add_paragraph(doc, text, bold=False, italic=False, spacing_after=6):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    p.paragraph_format.space_after = Pt(spacing_after)
    return p


def add_table(doc, headers, rows, style="Table Grid"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = style
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        run = hdr_cells[i].paragraphs[0].runs[0]
        run.bold = True
    for row_data in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row_data):
            cells[i].text = str(val)
    doc.add_paragraph()
    return table


def add_image_if_exists(doc, path, caption="", width=Inches(5.5)):
    p = Path(path)
    if p.exists():
        doc.add_picture(str(p), width=width)
        if caption:
            cap = doc.add_paragraph(caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].italic = True
    else:
        doc.add_paragraph(f"[Figure: {caption} — image not found at {path}]")


def create_report():
    doc = Document()

    # ── PAGE SETUP ──────────────────────────────────────────────────────────
    section = doc.sections[0]
    section.page_width  = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.5)
    section.top_margin  = section.bottom_margin = Cm(2.5)

    # ── COVER PAGE ──────────────────────────────────────────────────────────
    doc.add_paragraph()
    doc.add_paragraph()
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_p.add_run("Hotel Booking Analytics\n& Cancellation Prediction")
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = RGBColor(0x1F, 0x49, 0x8E)

    doc.add_paragraph()
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run("A Complete Data Science & Analytics Project Report").italic = True

    doc.add_paragraph()
    doc.add_paragraph()

    for label, value in [
        ("Student:", "Piyush Raikwar"),
        ("Program:", "B.Tech – Artificial Intelligence & Data Science"),
        ("Subject:", "Data Science & Machine Learning"),
        ("Dataset:", "Hotel Booking Demand (Kaggle)"),
        ("Tools:", "Python | Pandas | Scikit-learn | Streamlit"),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p.add_run(f"{label}  ")
        r1.bold = True
        p.add_run(value)

    doc.add_page_break()

    # ── ABSTRACT ────────────────────────────────────────────────────────────
    add_heading(doc, "Abstract", 1)
    add_paragraph(doc, (
        "This project presents a comprehensive data science analysis of hotel booking data "
        "from a publicly available dataset containing 119,390 records across 32 features. "
        "The analysis covers the complete data science workflow: data understanding, data cleaning, "
        "exploratory data analysis, feature engineering, machine learning model development, and "
        "interactive dashboard deployment. "
        "The primary business objective is to predict whether a hotel booking will be cancelled, "
        "enabling hotels to optimise revenue management and overbooking strategies. "
        "Four machine learning models were trained and compared. The best-performing model — "
        "Gradient Boosting — achieved a ROC-AUC of 0.8623 and an accuracy of 81.78% on the held-out "
        "test set. An interactive Streamlit dashboard was developed to present findings and allow "
        "real-time cancellation prediction."
    ))

    # ── 1. INTRODUCTION ─────────────────────────────────────────────────────
    add_heading(doc, "1. Introduction", 1)
    add_paragraph(doc, (
        "The hotel and hospitality industry faces significant challenges from booking cancellations. "
        "Cancellations disrupt revenue planning, lead to revenue loss when rooms cannot be resold, "
        "and create operational uncertainty. With the rise of online travel agencies (OTAs) offering "
        "flexible booking and free cancellation policies, cancellation rates have increased significantly. "
        "Data-driven approaches to predicting and managing cancellations offer a practical solution."
    ))
    add_paragraph(doc, (
        "This project leverages the Hotel Booking Demand dataset to build a complete analytical "
        "and predictive system. The work combines exploratory data analysis, statistical investigation "
        "of cancellation drivers, and machine learning-based cancellation prediction."
    ))

    # ── 2. PROBLEM STATEMENT ────────────────────────────────────────────────
    add_heading(doc, "2. Problem Statement", 1)
    add_paragraph(doc, (
        "Analyse hotel booking data to discover important booking, customer, pricing, and cancellation "
        "patterns, and develop a machine learning model that predicts whether a hotel booking is likely "
        "to be cancelled."
    ))

    # ── 3. OBJECTIVES ───────────────────────────────────────────────────────
    add_heading(doc, "3. Objectives", 1)
    objectives = [
        "Inspect, clean, and validate the raw hotel booking dataset.",
        "Perform comprehensive exploratory data analysis answering 24+ business questions.",
        "Discover patterns associated with booking cancellations.",
        "Engineer meaningful features from the raw dataset.",
        "Train and compare multiple machine learning classification models.",
        "Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.",
        "Identify the most influential features driving cancellation prediction.",
        "Build and deploy an interactive Streamlit dashboard for non-technical stakeholders.",
        "Prevent data leakage by excluding post-outcome variables from model training.",
        "Produce a reproducible, GitHub-ready project suitable for portfolio presentation.",
    ]
    for i, obj in enumerate(objectives, 1):
        doc.add_paragraph(f"{i}. {obj}")
    doc.add_paragraph()

    # ── 4. DATASET DESCRIPTION ──────────────────────────────────────────────
    add_heading(doc, "4. Dataset Description", 1)
    add_paragraph(doc, (
        "The dataset used is the Hotel Booking Demand dataset, publicly available on Kaggle at "
        "https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand."
    ))
    add_table(doc,
        ["Property", "Value"],
        [
            ["File name", "hotel_bookings.csv"],
            ["Raw rows", "119,390"],
            ["Raw columns", "32"],
            ["Cleaned rows", "87,230"],
            ["Date range", "July 2015 – August 2017"],
            ["Hotel types", "City Hotel, Resort Hotel"],
            ["Target variable", "is_canceled (0=Not Cancelled, 1=Cancelled)"],
            ["Cancellation rate (raw)", "~37.0%"],
            ["Cancellation rate (cleaned)", "~27.5%"],
        ]
    )

    add_heading(doc, "4.1 Key Features", 2)
    add_table(doc,
        ["Feature", "Description", "Type"],
        [
            ["hotel", "Hotel type: City Hotel or Resort Hotel", "Categorical"],
            ["is_canceled", "Target: 1 if cancelled, 0 if not", "Binary"],
            ["lead_time", "Days between booking and arrival date", "Numerical"],
            ["arrival_date_month", "Month of arrival", "Categorical"],
            ["adr", "Average Daily Rate (price per night in EUR)", "Numerical"],
            ["market_segment", "Booking channel", "Categorical"],
            ["deposit_type", "No Deposit / Non Refund / Refundable", "Categorical"],
            ["customer_type", "Transient / Contract / Group / Transient-Party", "Categorical"],
            ["previous_cancellations", "Number of previous cancellations by the guest", "Numerical"],
            ["total_of_special_requests", "Number of special requests made", "Numerical"],
            ["is_repeated_guest", "1 if repeat guest, 0 if new", "Binary"],
            ["reservation_status", "Check-Out / Canceled / No-Show (EXCLUDED - leakage)", "Categorical"],
        ]
    )

    # ── 5. TOOLS AND TECHNOLOGIES ───────────────────────────────────────────
    add_heading(doc, "5. Tools and Technologies", 1)
    add_table(doc,
        ["Tool / Library", "Version", "Purpose"],
        [
            ["Python", "3.10+", "Core programming language"],
            ["Pandas", "3.0+", "Data manipulation and analysis"],
            ["NumPy", "2.0+", "Numerical computation"],
            ["Matplotlib", "3.7+", "Static visualisation"],
            ["Seaborn", "0.13+", "Statistical visualisation"],
            ["Scikit-learn", "1.3+", "Machine learning models and preprocessing"],
            ["Joblib", "1.3+", "Model persistence"],
            ["Streamlit", "1.28+", "Interactive dashboard"],
            ["python-docx", "1.1+", "Report generation"],
            ["Git / GitHub", "—", "Version control and project hosting"],
        ]
    )

    # ── 6. METHODOLOGY ──────────────────────────────────────────────────────
    add_heading(doc, "6. Methodology", 1)
    add_paragraph(doc, "The project follows the standard data science process (CRISP-DM):")
    steps = [
        ("Business Understanding", "Defined the cancellation prediction problem and business value."),
        ("Data Understanding", "Inspected raw data: 119,390 rows, 32 columns, missing values, duplicates, suspicious records."),
        ("Data Preparation", "Removed 31,994 duplicate rows; imputed missing values; corrected invalid ADR; removed zero-guest records."),
        ("EDA", "Answered 24+ business questions; created 17 visualisations covering booking, pricing, and cancellation patterns."),
        ("Feature Engineering", "Created 10 new features including total_stay_nights, cancellation_history_rate, room_type_match."),
        ("Leakage Prevention", "Excluded reservation_status and reservation_status_date from all ML features."),
        ("Modelling", "Trained Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting."),
        ("Evaluation", "Compared models on Accuracy, Precision, Recall, F1, and ROC-AUC."),
        ("Deployment", "Built a 4-page Streamlit dashboard with live prediction capability."),
    ]
    for step, desc in steps:
        p = doc.add_paragraph()
        run = p.add_run(f"{step}: ")
        run.bold = True
        p.add_run(desc)
    doc.add_paragraph()

    # ── 7. DATA UNDERSTANDING ───────────────────────────────────────────────
    add_heading(doc, "7. Data Understanding", 1)
    add_heading(doc, "7.1 Missing Values", 2)
    add_table(doc,
        ["Column", "Missing Count", "Missing %", "Action"],
        [
            ["children", "4", "0.003%", "Fill with 0 (no children)"],
            ["country", "488", "0.41%", "Fill with 'Unknown'"],
            ["agent", "16,340", "13.69%", "Fill with 0 (no agent)"],
            ["company", "112,593", "94.31%", "Fill with 0 (no company)"],
        ]
    )

    add_heading(doc, "7.2 Duplicate Records", 2)
    add_paragraph(doc, (
        "31,994 exact duplicate rows were identified (26.8% of the dataset). "
        "These represent repeated data entry errors and were removed. "
        "After removing duplicates: 87,396 rows remain before further cleaning."
    ))

    add_heading(doc, "7.3 Suspicious / Invalid Values", 2)
    add_table(doc,
        ["Issue", "Count", "Action"],
        [
            ["ADR < 0 (negative price)", "1", "Replace with median ADR"],
            ["ADR > 5000 (extreme outlier)", "1", "Cap at 5000"],
            ["Zero-guest records (adults=0, children=0, babies=0)", "166", "Remove (invalid booking)"],
            ["Meal = 'Undefined'", "492", "Map to 'SC' (self-catering)"],
        ]
    )

    # ── 8. EDA FINDINGS ─────────────────────────────────────────────────────
    add_heading(doc, "8. Exploratory Data Analysis Findings", 1)

    add_heading(doc, "8.1 Booking Overview", 2)
    add_table(doc,
        ["Question", "Finding"],
        [
            ["Total bookings (cleaned)", "87,230"],
            ["Cancellation rate (cleaned)", "27.5%"],
            ["City Hotel share", "~64%"],
            ["Resort Hotel share", "~36%"],
            ["Peak booking month", "August (11,242 bookings)"],
            ["Lowest booking month", "January (4,685 bookings)"],
            ["Average lead time", "~80 days"],
            ["Average total stay", "3.6 nights"],
            ["Average ADR", "EUR 106.51"],
        ]
    )

    add_heading(doc, "8.2 Cancellation Patterns", 2)
    add_table(doc,
        ["Factor", "Observation"],
        [
            ["Lead time", "Cancelled bookings: avg 106 days; not cancelled: avg 70 days"],
            ["Deposit type", "Non-refundable deposits show very high cancellation rate"],
            ["Repeat guests", "Repeat guests cancel at 7.7% vs 28.3% for new guests"],
            ["Special requests", "More requests associated with lower cancellation rate"],
            ["Customer type", "Transient-Party shows the highest cancellation rate among types"],
            ["Previous cancellations", "Guests with cancellation history cancel current bookings at much higher rates"],
        ]
    )

    add_image_if_exists(doc, "images/01_cancellation_distribution.png",
                        "Figure 1: Cancellation Distribution")
    add_image_if_exists(doc, "images/05_lead_time_distribution.png",
                        "Figure 2: Lead Time - Cancelled vs Not Cancelled")
    add_image_if_exists(doc, "images/08_deposit_type_cancellation.png",
                        "Figure 3: Cancellation Rate by Deposit Type")

    # ── 9. FEATURE ENGINEERING ──────────────────────────────────────────────
    add_heading(doc, "9. Feature Engineering", 1)
    add_paragraph(doc, "Ten new features were created to improve predictive power:")
    add_table(doc,
        ["Feature", "Formula / Logic", "Rationale"],
        [
            ["total_stay_nights", "weekend_nights + week_nights", "Single measure of stay length"],
            ["total_guests", "adults + children + babies", "Group size as a predictor"],
            ["is_family", "1 if children>0 or babies>0", "Family bookings may behave differently"],
            ["is_weekend_booking", "1 if weekend_nights>0", "Weekend bookings vs weekday-only"],
            ["has_company", "1 if company_id != 0", "Corporate vs leisure"],
            ["has_agent", "1 if agent_id != 0", "OTA vs direct booking"],
            ["cancellation_history_rate", "prev_cancel / total_prev", "Historical cancel tendency"],
            ["room_type_match", "1 if reserved == assigned room", "Room satisfaction proxy"],
            ["lead_time_bin", "Categorical bucket of lead time", "Non-linear lead time effects"],
            ["adr_per_night", "ADR with zero-night guard", "Price signal with validity check"],
        ]
    )

    # ── 10. MACHINE LEARNING ────────────────────────────────────────────────
    add_heading(doc, "10. Machine Learning Methodology", 1)

    add_heading(doc, "10.1 Data Leakage Prevention", 2)
    add_paragraph(doc, "The following columns were excluded from all ML features:")
    add_table(doc,
        ["Excluded Column", "Reason"],
        [
            ["reservation_status", "Directly encodes the cancellation outcome (Check-Out, Canceled, No-Show)"],
            ["reservation_status_date", "Post-outcome date — unavailable at prediction time"],
            ["is_canceled", "The target variable itself"],
            ["arrival_date_year", "2015-2017 span; year not generalisable"],
            ["arrival_date_week_number", "Too granular; year + month sufficient"],
            ["arrival_date_day_of_month", "Too granular for booking-time prediction"],
        ]
    )

    add_heading(doc, "10.2 Preprocessing Pipeline", 2)
    add_paragraph(doc, (
        "A scikit-learn ColumnTransformer pipeline was built with two sub-pipelines:\n"
        "1. Numerical features: SimpleImputer (strategy=median) → StandardScaler\n"
        "2. Categorical features: SimpleImputer (strategy=constant, 'Unknown') → OneHotEncoder (drop='first')"
    ))

    add_heading(doc, "10.3 Train/Test Split", 2)
    add_paragraph(doc, (
        "80/20 stratified train/test split with random_state=42 to ensure reproducibility.\n"
        "Training set: 69,784 records | Test set: 17,446 records\n"
        "Stratification ensures the same cancellation rate in both splits (~27.5%)."
    ))

    add_heading(doc, "10.4 Models Trained", 2)
    add_table(doc,
        ["Model", "Key Parameters"],
        [
            ["Logistic Regression", "max_iter=1000, class_weight='balanced'"],
            ["Decision Tree", "max_depth=10, class_weight='balanced'"],
            ["Random Forest", "n_estimators=200, max_depth=15, class_weight='balanced'"],
            ["Gradient Boosting", "n_estimators=200, max_depth=5, learning_rate=0.1"],
        ]
    )

    # ── 11. MODEL EVALUATION ────────────────────────────────────────────────
    add_heading(doc, "11. Model Evaluation Results", 1)

    add_heading(doc, "11.1 Model Comparison Table", 2)
    add_table(doc,
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        [
            ["Gradient Boosting ★", "0.8178", "0.7365", "0.5267", "0.6141", "0.8623"],
            ["Random Forest", "0.7484", "0.5278", "0.8140", "0.6404", "0.8577"],
            ["Decision Tree", "0.7076", "0.4822", "0.8444", "0.6138", "0.8387"],
            ["Logistic Regression", "0.7207", "0.4954", "0.7876", "0.6082", "0.8341"],
        ]
    )

    add_heading(doc, "11.2 Best Model: Gradient Boosting", 2)
    add_paragraph(doc, "Selected by highest ROC-AUC score (0.8623).")
    add_paragraph(doc, "Classification Report (test set):", bold=True)
    add_table(doc,
        ["Class", "Precision", "Recall", "F1-Score", "Support"],
        [
            ["Not Cancelled (0)", "0.84", "0.93", "0.88", "12,644"],
            ["Cancelled (1)", "0.74", "0.53", "0.61", "4,802"],
            ["Macro Average", "0.79", "0.73", "0.75", "17,446"],
            ["Weighted Average", "0.81", "0.82", "0.81", "17,446"],
        ]
    )

    add_heading(doc, "11.3 Why ROC-AUC is the Primary Metric", 2)
    add_paragraph(doc, (
        "ROC-AUC was chosen as the primary evaluation metric because:\n"
        "1. It is threshold-independent, evaluating model performance across all possible classification thresholds.\n"
        "2. It handles class imbalance better than accuracy alone.\n"
        "3. Hotels care about both false positives (incorrectly flagging legitimate guests) and "
        "false negatives (missing real cancellations), and ROC-AUC captures this trade-off."
    ))

    add_image_if_exists(doc, "images/15_roc_curves.png", "Figure 4: ROC Curves – All Models")
    add_image_if_exists(doc, "images/14_confusion_matrices.png", "Figure 5: Confusion Matrices – All Models")

    # ── 12. FEATURE IMPORTANCE ──────────────────────────────────────────────
    add_heading(doc, "12. Feature Importance / Model Interpretability", 1)
    add_paragraph(doc, (
        "The Gradient Boosting model provides built-in feature importance scores "
        "based on how much each feature reduces impurity across all trees. "
        "The most influential features (after one-hot encoding) are:"
    ))
    add_table(doc,
        ["Rank", "Feature", "Interpretation"],
        [
            ["1", "deposit_type_Non Refund", "Non-refundable deposits strongly associated with cancellation"],
            ["2", "lead_time", "Longer booking lead time -> higher cancellation risk"],
            ["3", "arrival_month_num", "Seasonality affects cancellation patterns"],
            ["4", "previous_cancellations", "Historical cancellation behaviour predicts future cancellations"],
            ["5", "total_of_special_requests", "More requests -> lower cancellation probability"],
            ["6", "cancellation_history_rate", "Guest's personal cancellation ratio"],
            ["7", "adr", "Price sensitivity influences cancellation decisions"],
            ["8", "days_in_waiting_list", "Wait-listed guests may cancel once alternatives found"],
        ]
    )
    add_image_if_exists(doc, "images/17_feature_importance.png",
                        "Figure 6: Top 15 Feature Importances – Gradient Boosting")

    # ── 13. DASHBOARD ───────────────────────────────────────────────────────
    add_heading(doc, "13. Interactive Dashboard", 1)
    add_paragraph(doc, (
        "A four-page Streamlit dashboard (dashboard/app.py) was developed to present "
        "all findings in an accessible, interactive format for non-technical stakeholders."
    ))
    add_table(doc,
        ["Page", "Content"],
        [
            ["Page 1: Executive Overview", "8 KPI metrics, cancellation distribution chart, quick summary table"],
            ["Page 2: Booking Analytics", "Monthly demand trend, hotel comparison, customer types, market segments, ADR trends"],
            ["Page 3: Cancellation Analytics", "Cancellation by deposit/customer/segment/lead time/special requests"],
            ["Page 4: ML Prediction", "Interactive form with 15+ input fields; returns cancellation class + probability"],
        ]
    )
    add_paragraph(doc, "Run with: streamlit run dashboard/app.py")

    # ── 14. RESULTS ─────────────────────────────────────────────────────────
    add_heading(doc, "14. Results Summary", 1)
    add_table(doc,
        ["Aspect", "Result"],
        [
            ["Best model", "Gradient Boosting"],
            ["ROC-AUC", "0.8623"],
            ["Test accuracy", "81.78%"],
            ["Precision (Cancelled)", "73.65%"],
            ["Recall (Cancelled)", "52.67%"],
            ["F1 (Cancelled)", "61.41%"],
            ["Most important feature", "deposit_type_Non Refund"],
            ["Cleaned dataset size", "87,230 records"],
            ["Training samples", "69,784"],
            ["Test samples", "17,446"],
        ]
    )

    # ── 15. CHALLENGES & LIMITATIONS ────────────────────────────────────────
    add_heading(doc, "15. Challenges and Limitations", 1)
    challenges = [
        "High duplicate rate (26.8%) required careful removal to avoid data distortion.",
        "The 'company' column had 94% missing values — filled with 0 (no company), which may underrepresent the corporate segment.",
        "Non-refundable deposit types show near-100% cancellation rates, which may reflect a data collection artefact.",
        "The model achieves lower recall (53%) for cancellations — some actual cancellations are missed.",
        "The dataset spans only 3 years (2015-2017) from a limited geographic area; generalisation may be limited.",
        "Seasonal patterns in the data (COVID impact, post-pandemic behaviour) are not captured.",
    ]
    for c in challenges:
        doc.add_paragraph(c, style="List Bullet")
    doc.add_paragraph()

    # ── 16. FUTURE SCOPE ────────────────────────────────────────────────────
    add_heading(doc, "16. Future Scope", 1)
    future = [
        "Hyperparameter tuning using GridSearchCV or RandomizedSearchCV to further optimise model performance.",
        "SHAP (SHapley Additive exPlanations) for per-prediction interpretability — showing WHY each prediction was made.",
        "Class balancing techniques (SMOTE, adjusted weights) to improve recall for the minority (cancelled) class.",
        "Temporal cross-validation to simulate real-world deployment where the model is trained on past data and tested on future.",
        "FastAPI REST endpoint for real-world integration with Property Management Systems (PMS).",
        "Multi-hotel generalisation — testing the model on data from different regions and hotel chains.",
    ]
    for f in future:
        doc.add_paragraph(f, style="List Bullet")
    doc.add_paragraph()

    # ── 17. CONCLUSION ──────────────────────────────────────────────────────
    add_heading(doc, "17. Conclusion", 1)
    add_paragraph(doc, (
        "This project successfully demonstrates a complete end-to-end data science workflow applied to "
        "a real-world hotel booking dataset. Starting from raw, uncleaned data with 31,994 duplicate records "
        "and several data quality issues, the project produced a clean, analysis-ready dataset, performed "
        "comprehensive exploratory data analysis, and trained multiple machine learning models."
    ))
    add_paragraph(doc, (
        "The best-performing model — Gradient Boosting — achieves a ROC-AUC of 0.8623, indicating strong "
        "discriminative ability. Key drivers of cancellation include deposit type (non-refundable), lead time, "
        "and historical cancellation behaviour. These findings align with domain knowledge and provide "
        "actionable insights for hotel revenue management."
    ))
    add_paragraph(doc, (
        "The project is fully reproducible, GitHub-ready, and includes an interactive Streamlit dashboard "
        "for business-user exploration and real-time cancellation prediction."
    ))

    # ── 18. REFERENCES ──────────────────────────────────────────────────────
    add_heading(doc, "18. References", 1)
    refs = [
        "Mostipak, J. (2020). Hotel Booking Demand. Kaggle. https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand",
        "Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. Data in Brief, 22, 41-49.",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.",
        "McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proc. of the 9th Python in Science Conference.",
        "Streamlit Inc. (2023). Streamlit – The fastest way to build data apps. https://streamlit.io",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
        "Friedman, J.H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189-1232.",
    ]
    for ref in refs:
        doc.add_paragraph(ref, style="List Number")

    # ── SAVE ────────────────────────────────────────────────────────────────
    out_path = Path("Piyush_Hotel_Booking_Project_Report.docx")
    doc.save(str(out_path))
    print(f"Report saved: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    create_report()
