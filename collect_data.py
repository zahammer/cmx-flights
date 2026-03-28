"""
collect_data.py
===============
Downloads historical flight data from BTS and weather data from Open-Meteo
for CMX (Houghton) and ORD (Chicago O'Hare), then merges them into a
single CSV for model training.

Run:  python collect_data.py
Output: data/flights_merged.csv
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

os.makedirs("data", exist_ok=True)

# ──────────────────────────────────────────────
# STEP 1: BTS Flight Data
# ──────────────────────────────────────────────
# BTS (Bureau of Transportation Statistics) provides free CSVs of all
# US domestic flights including on-time performance.
#
# Manual download (easiest):
#   1. Go to https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ
#   2. Select fields: Year, Month, DayofMonth, FlightDate, Reporting_Airline,
#      Flight_Number_Reporting_Airline, Origin, Dest, DepDelay, ArrDelay,
#      Cancelled, Diverted
#   3. Filter: Origin = CMX  OR  Dest = CMX
#   4. Download the CSV and place it at: data/bts_raw.csv
#
# This script will then process it. If you have the file, continue:

BTS_FILE = "data/bts_raw.csv"

def load_bts():
    if not os.path.exists(BTS_FILE):
        print("⚠  BTS file not found at data/bts_raw.csv")
        print("   Please download it manually from:")
        print("   https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ")
        print("   Filter for: Origin=CMX or Dest=CMX, Year 2023-2024")
        print("   Then re-run this script.")
        return None

    print("📂 Loading BTS flight data...")
    df = pd.read_csv(BTS_FILE, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Keep only CMX flights
    df = df[(df['ORIGIN'] == 'CMX') | (df['DEST'] == 'CMX')].copy()

    # Parse date
    if 'FL_DATE' in df.columns:
        df['date'] = pd.to_datetime(df['FL_DATE'])
    elif 'FlightDate' in df.columns:
        df['date'] = pd.to_datetime(df['FlightDate'])

    df['cancelled']   = df.get('CANCELLED', df.get('Cancelled', 0)).fillna(0).astype(int)
    df['dep_delay']   = df.get('DEP_DELAY', df.get('DepDelay',   0)).fillna(0).astype(float)
    df['delayed']     = (df['dep_delay'] > 15).astype(int)
    df['origin']      = df.get('ORIGIN', df.get('Origin', ''))
    df['dest']        = df.get('DEST',   df.get('Dest',   ''))
    df['flight_date'] = df['date'].dt.date.astype(str)

    print(f"   ✓ {len(df)} CMX flights loaded")
    return df[['flight_date','origin','dest','cancelled','dep_delay','delayed']]


# ──────────────────────────────────────────────
# STEP 2: Historical Weather from Open-Meteo
# ──────────────────────────────────────────────
def fetch_weather_history(lat, lon, start, end, label):
    """Fetch daily weather summaries for a location between start and end dates."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&daily=weather_code,wind_speed_10m_max,snowfall_sum,precipitation_sum,visibility_mean"
        "&wind_speed_unit=mph&timezone=America%2FChicago"
    )
    print(f"   Fetching {label} weather {start} → {end}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()['daily']
    df = pd.DataFrame(data)
    df.rename(columns={
        'time': 'flight_date',
        'weather_code': f'wmo_{label}',
        'wind_speed_10m_max': f'wind_{label}',
        'snowfall_sum': f'snow_{label}',
        'precipitation_sum': f'precip_{label}',
        'visibility_mean': f'vis_{label}',
    }, inplace=True)
    return df


def load_weather(start_date, end_date):
    cmx_file = "data/weather_cmx.csv"
    ord_file  = "data/weather_ord.csv"

    if not os.path.exists(cmx_file):
        cmx = fetch_weather_history(47.1684, -88.4891, start_date, end_date, "cmx")
        cmx.to_csv(cmx_file, index=False)
        time.sleep(1)
    else:
        print("   ✓ CMX weather loaded from cache")
        cmx = pd.read_csv(cmx_file)

    if not os.path.exists(ord_file):
        ord_ = fetch_weather_history(41.9742, -87.9073, start_date, end_date, "ord")
        ord_.to_csv(ord_file, index=False)
    else:
        print("   ✓ ORD weather loaded from cache")
        ord_ = pd.read_csv(ord_file)

    return cmx, ord_


# ──────────────────────────────────────────────
# STEP 3: Merge
# ──────────────────────────────────────────────
def merge_and_save(flights, cmx_w, ord_w):
    df = flights.copy()
    df = df.merge(cmx_w, on='flight_date', how='left')
    df = df.merge(ord_w,  on='flight_date', how='left')

    # Extra features
    df['date']        = pd.to_datetime(df['flight_date'])
    df['month']       = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek   # 0=Mon … 6=Sun
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['is_winter']   = df['month'].isin([12, 1, 2, 3]).astype(int)
    df['cmx_snow_flag'] = (df['snow_cmx'] > 0.1).astype(int)
    df['ord_snow_flag'] = (df['snow_ord'] > 0.1).astype(int)

    out = "data/flights_merged.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Merged dataset saved → {out}")
    print(f"   Shape: {df.shape}")
    print(f"   Cancellation rate: {df['cancelled'].mean():.1%}")
    print(f"   Delay rate (>15 min): {df['delayed'].mean():.1%}")
    return df


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    START = "2023-01-01"
    END   = datetime.today().strftime("%Y-%m-%d")

    print("=== CMX AI Data Collection ===\n")

    flights = load_bts()
    if flights is None:
        print("\n⏸  Fix the BTS download above, then re-run.")
        exit(1)

    # Determine actual date range in the flight data
    start_actual = flights['flight_date'].min()
    end_actual   = flights['flight_date'].max()
    print(f"   Date range in BTS data: {start_actual} → {end_actual}")

    print("\n📡 Fetching historical weather...")
    cmx_w, ord_w = load_weather(start_actual, end_actual)

    print("\n🔗 Merging datasets...")
    df = merge_and_save(flights, cmx_w, ord_w)

    print("\nFirst 5 rows:")
    print(df[['flight_date','cancelled','delayed','wind_cmx','snow_cmx','wind_ord','snow_ord']].head())
