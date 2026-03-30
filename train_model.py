"""
train_model.py  —  CMX Delay Prediction: Model Training & Evaluation
=====================================================================
Trains two models to predict whether a CMX flight will be delayed >15 min:
  - Baseline: Logistic Regression
  - Improved: Random Forest

Outputs:
  - model.pkl          (saved model for serving predictions)
  - results/report.txt (accuracy metrics for your project report)
  - results/feature_importance.csv

Run AFTER collect_data.py:
    pip install scikit-learn pandas joblib matplotlib
    python train_model.py
"""

import os, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FEATURES used to predict delay
# ─────────────────────────────────────────────────────────────
FEATURES = [
    'wind_cmx',       # Max wind speed at CMX (mph)
    'gusts_cmx',      # Max wind gusts at CMX (mph)
    'snow_cmx',       # Total snowfall at CMX (inches)
    'precip_cmx',     # Total precipitation at CMX
    'wind_ord',       # Max wind speed at ORD (mph)
    'gusts_ord',      # Max wind gusts at ORD (mph)
    'snow_ord',       # Total snowfall at ORD (inches)
    'precip_ord',     # Total precipitation at ORD
    'snow_flag_cmx',  # Binary: snow at CMX
    'snow_flag_ord',  # Binary: snow at ORD
    'wind_flag_cmx',  # Binary: high wind at CMX
    'wind_flag_ord',  # Binary: high wind at ORD
    'storm_flag_cmx', # Binary: thunderstorm at CMX
    'storm_flag_ord', # Binary: thunderstorm at ORD
    'month',          # Month (1–12)
    'day_of_week',    # Day of week (0=Mon … 6=Sun)
    'is_winter',      # Dec–Mar flag
]
TARGET = 'delayed'


def load_data():
    path = "data/flights_merged.csv"
    if not os.path.exists(path):
        print("ERROR: data/flights_merged.csv not found.")
        print("Run collect_data.py first.")
        exit(1)

    df = pd.read_csv(path)

    # Drop rows missing key features or target
    df = df.dropna(subset=[TARGET] + [f for f in FEATURES if f in df.columns])

    # Fill any remaining NaN in features with 0
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    print(f"Loaded {len(df)} rows")
    print(f"Delay rate: {df[TARGET].mean():.1%}")
    return df


def evaluate_model(name, model, X_train, X_test, y_train, y_test, report_lines):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc   = roc_auc_score(y_test, y_prob)
    cv    = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5), scoring='f1').mean()
    report = classification_report(y_test, y_pred, target_names=['On Time', 'Delayed'])

    lines = [
        f"\n{'='*55}",
        f"MODEL: {name}",
        f"{'='*55}",
        f"Cross-val F1 (train): {cv:.3f}",
        f"ROC-AUC (test):       {auc:.3f}",
        "",
        "Classification Report (test set):",
        report,
    ]
    for l in lines:
        print(l)
    report_lines.extend(lines)
    return model, auc, y_prob


if __name__ == "__main__":
    print("CMX Delay Prediction — Model Training\n")

    df = load_data()
    X  = df[FEATURES].values
    y  = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows\n")

    report_lines = [
        "CMX FLIGHT DELAY PREDICTION — MODEL EVALUATION REPORT",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Dataset:   {len(df)} CMX flights  |  Delay rate: {y.mean():.1%}",
        f"Features:  {len(FEATURES)}",
    ]

    # ── Baseline: Logistic Regression ──────────────────────
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    lr_model, lr_auc, lr_prob = evaluate_model(
        "Baseline — Logistic Regression", lr,
        X_train, X_test, y_train, y_test, report_lines
    )

    # ── Improved: Random Forest ─────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_model, rf_auc, rf_prob = evaluate_model(
        "Improved — Random Forest", rf,
        X_train, X_test, y_train, y_test, report_lines
    )

    # ── Feature Importance ──────────────────────────────────
    importance = pd.Series(rf.feature_importances_, index=FEATURES)\
        .sort_values(ascending=False)
    importance.to_csv("results/feature_importance.csv", header=['importance'])

    imp_lines = [
        "\nTop Feature Importances (Random Forest):",
        importance.head(10).to_string(),
        f"\nBaseline AUC: {lr_auc:.3f}",
        f"Improved AUC: {rf_auc:.3f}",
        f"Improvement:  +{rf_auc - lr_auc:.3f}",
    ]
    for l in imp_lines:
        print(l)
    report_lines.extend(imp_lines)

    # ── Save report ─────────────────────────────────────────
    with open("results/report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print("\nReport saved → results/report.txt")

    # ── Save best model ─────────────────────────────────────
    joblib.dump({
        'model':    rf,
        'features': FEATURES,
        'auc':      rf_auc,
    }, 'model.pkl')
    print("Model saved → model.pkl")

    # ── Plot feature importance ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.head(10).plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('CMX Delay Prediction — Top 10 Features')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)
    print("Plot saved → results/feature_importance.png")

    # ── ROC Curve ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, lr_prob, name="Logistic Regression", ax=ax)
    RocCurveDisplay.from_predictions(y_test, rf_prob, name="Random Forest", ax=ax)
    ax.set_title("ROC Curve — CMX Delay Prediction")
    plt.tight_layout()
    plt.savefig("results/roc_curve.png", dpi=150)
    print("ROC curve saved → results/roc_curve.png")

    print("\nDone! Check the results/ folder for charts and your report.txt.")
