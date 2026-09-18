"""
model_training.py
-----------------
Hotel Booking Analytics & Cancellation Prediction
Piyush Raikwar | B.Tech AI & Data Science

Trains, evaluates, and saves the cancellation prediction models.
Includes Logistic Regression, Decision Tree, Random Forest, and
Gradient Boosting for comparison.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
)

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_cleaning import load_raw_data, clean_data
from src.feature_engineering import engineer_features
from src.preprocessing import build_preprocessor, get_X_y

RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_DIR = Path("models")


def split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def build_model_pipelines(preprocessor) -> dict:
    """Return a dict of named model pipelines."""
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
            ))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                max_depth=10, random_state=RANDOM_STATE, class_weight="balanced"
            ))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=RANDOM_STATE,
                n_jobs=-1, class_weight="balanced"
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=RANDOM_STATE
            ))
        ]),
    }


def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return evaluation metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }


def train_and_evaluate(
    save_best: bool = True,
    verbose: bool = True
) -> tuple[dict, dict, dict]:
    """
    Full training pipeline.
    Returns:
        results     – dict of model_name → metrics dict
        pipelines   – dict of model_name → fitted Pipeline
        data_splits – dict with X_train, X_test, y_train, y_test
    """
    # ── Load & prepare data ───────────────────────────────────────────────
    raw = load_raw_data()
    clean = clean_data(raw, verbose=verbose)
    fe = engineer_features(clean, verbose=verbose)
    X, y = get_X_y(fe)

    X_train, X_test, y_train, y_test = split_data(X, y)
    if verbose:
        print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")
        print(f"Cancellation rate – train: {y_train.mean():.3f}  test: {y_test.mean():.3f}")

    preprocessor = build_preprocessor()
    pipelines = build_model_pipelines(preprocessor)
    results = {}

    for name, pipeline in pipelines.items():
        if verbose:
            print(f"\n[Training] {name} …")
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics
        if verbose:
            print(f"  Accuracy : {metrics['Accuracy']:.4f}")
            print(f"  Precision: {metrics['Precision']:.4f}")
            print(f"  Recall   : {metrics['Recall']:.4f}")
            print(f"  F1       : {metrics['F1']:.4f}")
            print(f"  ROC-AUC  : {metrics['ROC-AUC']:.4f}")

    # ── Select best model by ROC-AUC ──────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["ROC-AUC"])
    if verbose:
        print(f"\n[Best Model] {best_name}  ROC-AUC = {results[best_name]['ROC-AUC']:.4f}")

    # ── Save best model ───────────────────────────────────────────────────
    if save_best:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(pipelines[best_name], MODEL_DIR / "model.pkl")
        joblib.dump(best_name, MODEL_DIR / "best_model_name.pkl")
        # Save feature column list for the dashboard
        joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.pkl")
        if verbose:
            print(f"[Saved] models/model.pkl  (pipeline including preprocessor)")

    data_splits = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }
    return results, pipelines, data_splits


def get_feature_importance(pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature importances from a tree-based model pipeline."""
    clf = pipeline.named_steps["classifier"]
    pre = pipeline.named_steps["preprocessor"]

    # Get feature names from ColumnTransformer
    try:
        encoded_names = pre.get_feature_names_out()
    except Exception:
        encoded_names = [f"feature_{i}" for i in range(len(clf.feature_importances_))]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return pd.DataFrame()

    return pd.DataFrame({
        "feature": encoded_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(20)


if __name__ == "__main__":
    results, pipelines, splits = train_and_evaluate(save_best=True, verbose=True)
    print("\n=== MODEL COMPARISON TABLE ===")
    comp = pd.DataFrame(results).T.round(4)
    print(comp)
