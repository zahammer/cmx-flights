"""
train_model.py
==============
Trains a cancellation and delay classifier using the merged dataset
from collect_data.py. Saves the model as model.pkl for use in
a Flask prediction server.

Run:  python train_model.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ──────────────────────────────────────────────
# FEATURES & TARGETS
# ──────────────────────────────────────────────
FEATURES = [
    'wind_cmx',       # Max wind speed at CMX (mph)
    'snow_cmx',       # Snowfall at CMX (inches)
    'precip_cmx',     # Precipitation at CMX
    'wind_ord',       # Max wind speed at ORD (mph)
    'snow_ord',       # Snowfall at ORD (inches)
    'precip_ord',     # Precipitation at ORD
    'cmx_snow_flag',  # Binary: snow at CMX
    'ord_snow_flag',  # Binary: snow at ORD
    'month',          # Month (1–12)
    'day_of_week',    # Day of week (0–6)
    'is_weekend',     # Weekend flag
    'is_winter',      # Dec–Mar flag
]

def load_data():
    df = pd.read_csv("data/flights_merged.csv")
    df = df.dropna(subset=FEATURES + ['cancelled', 'delayed'])
    print(f"✓ Loaded {len(df)} rows after dropping NaN rows")
    return df

def train_and_evaluate(df, target_col, target_name):
    print(f"\n{'='*50}")
    print(f"TARGET: {target_name}")
    print(f"{'='*50}")

    X = df[FEATURES].values
    y = df[target_col].values

    print(f"Class distribution: {np.bincount(y)} ({y.mean():.1%} positive)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Baseline: Logistic Regression ──
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42))
    ])
    lr_scores = cross_val_score(lr_pipe, X_train, y_train, cv=5, scoring='f1')
    print(f"\nBaseline (Logistic Regression) CV F1: {lr_scores.mean():.3f} ± {lr_scores.std():.3f}")

    # ── Improved: Random Forest ──
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
    print(f"Improved  (Random Forest)       CV F1: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")

    # ── Final evaluation ──
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(f"\nTest Set Report:\n{classification_report(y_test, y_pred)}")

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(8).to_string())

    return rf

def save_model(cancel_model, delay_model):
    bundle = {
        'cancel_model': cancel_model,
        'delay_model':  delay_model,
        'features':     FEATURES,
    }
    joblib.dump(bundle, 'model.pkl')
    print("\n✅ Model saved → model.pkl")

# ──────────────────────────────────────────────
# FLASK SERVER (run separately or append here)
# ──────────────────────────────────────────────
FLASK_SERVER = '''
# predict_server.py  — run with: python predict_server.py
# Then call from index.html: fetch('http://localhost:5000/predict?...')

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import requests

app = Flask(__name__)
CORS(app)

bundle = joblib.load('model.pkl')
cancel_model = bundle['cancel_model']
delay_model  = bundle['delay_model']
FEATURES     = bundle['features']

def get_current_weather(lat, lon):
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           f"&current=weather_code,wind_speed_10m,snowfall"
           f"&daily=snowfall_sum,precipitation_sum"
           f"&forecast_days=1&wind_speed_unit=mph")
    r = requests.get(url, timeout=10)
    d = r.json()
    return {
        'wind':   d['current'].get('wind_speed_10m', 0),
        'snow':   d['daily']['snowfall_sum'][0] or 0,
        'precip': d['daily']['precipitation_sum'][0] or 0,
        'wmo':    d['current'].get('weather_code', 0),
    }

@app.route('/predict')
def predict():
    from datetime import datetime
    now = datetime.now()
    cmx = get_current_weather(47.1684, -88.4891)
    ord_ = get_current_weather(41.9742, -87.9073)

    X = [[
        cmx['wind'], cmx['snow'], cmx['precip'],
        ord_['wind'], ord_['snow'], ord_['precip'],
        int(cmx['snow'] > 0.1), int(ord_['snow'] > 0.1),
        now.month, now.weekday(),
        int(now.weekday() >= 5),
        int(now.month in [12,1,2,3]),
    ]]

    cancel_prob = cancel_model.predict_proba(X)[0][1]
    delay_prob  = delay_model.predict_proba(X)[0][1]

    reasons = []
    if cmx['wind'] > 25: reasons.append(f"Strong CMX winds ({cmx['wind']:.0f} mph)")
    if ord_['wind'] > 25: reasons.append(f"Strong ORD winds ({ord_['wind']:.0f} mph)")
    if cmx['snow'] > 0.1: reasons.append("Snow at CMX")
    if ord_['snow'] > 0.1: reasons.append("Snow at ORD")
    if not reasons: reasons = ["Favorable conditions"]

    return jsonify({
        'cancel_pct': round(cancel_prob * 100),
        'delay_pct':  round(delay_prob  * 100),
        'reason':     ', '.join(reasons)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
'''

if __name__ == "__main__":
    print("=== CMX AI Model Training ===\n")

    df = load_data()

    cancel_model = train_and_evaluate(df, 'cancelled', 'CANCELLATION')
    delay_model  = train_and_evaluate(df, 'delayed',   'DELAY (>15 min)')

    save_model(cancel_model, delay_model)

    # Write Flask server file
    with open('predict_server.py', 'w') as f:
        f.write(FLASK_SERVER.strip())
    print("✅ Flask server written → predict_server.py")
    print("\nNext steps:")
    print("  1. pip install flask flask-cors")
    print("  2. python predict_server.py")
    print("  3. Update runAIPredictions() in index.html to call http://localhost:5000/predict")
    print("     Or deploy to Render/Railway for a live endpoint.")
