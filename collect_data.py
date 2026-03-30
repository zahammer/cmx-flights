"""
collect_data.py  —  CMX Delay Prediction: Data Collection
==========================================================
Downloads:
  1. Historical weather for CMX and ORD from Open-Meteo (free, no key)
  2. Merges with BTS flight data (manual download required — see instructions)

Run:
    pip install pandas requests
    python collect_data.py

Output:  data/flights_merged.csv
"""

import os, time, requests, pandas as pd
from datetime import datetime

os.makedirs("data", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1:  BTS Flight Data  (manual download)
# ─────────────────────────────────────────────────────────────
# 1. Go to:
#    https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ
#
# 2. Under "Filter Geography", set Origin = CMX
#
# 3. Select these fields:
#      Year, Month, DayofMonth, FlightDate
#      Reporting_Airline, Flight_Number_Reporting_Airline
#      Origin, Dest
#      DepDelay, ArrDelay, Cancelled, WeatherDelay
#
# 4. Download for years 2022, 2023, 2024 (one at a time)
#    Save as: data/bts_2022.csv, data/bts_2023.csv, data/bts_2024.csv
#
# ─────────────────────────────────────────────────────────────

BTS_FILES = ["data/bts_2022.csv", "data/bts_2023.csv", "data/bts_2024.csv"]

def load_bts():
    found = [f for f in BTS_FILES if os.path.exists(f)]
    if not found:
        print("=" * 60)
        print("BTS FILES NOT FOUND — Please download from:")
        print("https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ")
        print()
        print("Steps:")
        print("  1. Filter Origin = CMX")
        print("  2. Select: FlightDate, DepDelay, Cancelled, Origin, Dest")
        print("  3. Download 2022, 2023, 2024 separately")
        print("  4. Save to data/bts_2022.csv etc.")
        print("  5. Re-run this script")
        return None

    print(f"Loading BTS data from {len(found)} file(s)...")
    dfs = []
    for f in found:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    rename = {
        'FL_DATE': 'flight_date', 'FlightDate': 'flight_date',
        'ORIGIN': 'origin', 'Origin': 'origin',
        'DEST': 'dest', 'Dest': 'dest',
        'DEP_DELAY': 'dep_delay', 'DepDelay': 'dep_delay',
        'CANCELLED': 'cancelled', 'Cancelled': 'cancelled',
        'WEATHER_DELAY': 'weather_delay', 'WeatherDelay': 'weather_delay',
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    df = df[(df.get('origin', '') == 'CMX') | (df.get('dest', '') == 'CMX')].copy()
    df['flight_date'] = pd.to_datetime(df['flight_date'], errors='coerce')
    df['date_str']    = df['flight_date'].dt.strftime('%Y-%m-%d')
    df['dep_delay']   = pd.to_numeric(df.get('dep_delay', 0), errors='coerce').fillna(0)
    df['cancelled']   = pd.to_numeric(df.get('cancelled', 0), errors='coerce').fillna(0).astype(int)
    df['delayed']     = (df['dep_delay'] > 15).astype(int)
    df['month']       = df['flight_date'].dt.month
    df['day_of_week'] = df['flight_date'].dt.dayofweek
    df['is_winter']   = df['month'].isin([12, 1, 2, 3]).astype(int)

    print(f"  {len(df)} CMX flight records")
    print(f"  Date range: {df['date_str'].min()} to {df['date_str'].max()}")
    print(f"  Delay rate (>15 min): {df['delayed'].mean():.1%}")
    return df


def fetch_weather(lat, lon, start, end, label):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&daily=weather_code,wind_speed_10m_max,wind_gusts_10m_max,"
        "snowfall_sum,precipitation_sum,visibility_mean"
        "&wind_speed_unit=mph&timezone=America%2FChicago"
    )
    print(f"  Fetching {label} weather {start} to {end}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    d = r.json()['daily']
    return pd.DataFrame(d).rename(columns={
        'time': 'date_str',
        'weather_code': f'wmo_{label}',
        'wind_speed_10m_max': f'wind_{label}',
        'wind_gusts_10m_max': f'gusts_{label}',
        'snowfall_sum': f'snow_{label}',
        'precipitation_sum': f'precip_{label}',
        'visibility_mean': f'vis_{label}',
    })


def load_weather(start_date, end_date):
    cmx_file, ord_file = "data/weather_cmx.csv", "data/weather_ord.csv"
    if not os.path.exists(cmx_file):
        fetch_weather(47.1684, -88.4891, start_date, end_date, "cmx").to_csv(cmx_file, index=False)
        time.sleep(1)
    else:
        print("  CMX weather loaded from cache")
    if not os.path.exists(ord_file):
        fetch_weather(41.9742, -87.9073, start_date, end_date, "ord").to_csv(ord_file, index=False)
    else:
        print("  ORD weather loaded from cache")
    return pd.read_csv(cmx_file), pd.read_csv(ord_file)


if __name__ == "__main__":
    print("CMX Delay Prediction — Data Collection\n")

    flights = load_bts()
    if flights is None:
        exit(1)

    start, end = flights['date_str'].min(), flights['date_str'].max()
    print(f"\nFetching weather ({start} to {end})...")
    cmx_w, ord_w = load_weather(start, end)

    print("\nMerging...")
    df = flights.merge(cmx_w, on='date_str', how='left').merge(ord_w, on='date_str', how='left')
    df['snow_flag_cmx']  = (df['snow_cmx']  > 0.1).astype(int)
    df['snow_flag_ord']  = (df['snow_ord']   > 0.1).astype(int)
    df['wind_flag_cmx']  = (df['wind_cmx']   > 25).astype(int)
    df['wind_flag_ord']  = (df['wind_ord']   > 25).astype(int)
    df['storm_flag_cmx'] = (df['wmo_cmx'].isin([95,96,99])).astype(int)
    df['storm_flag_ord'] = (df['wmo_ord'].isin([95,96,99])).astype(int)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/flights_merged.csv", index=False)
    print(f"\nSaved: data/flights_merged.csv  ({df.shape[0]} rows)")
    print("\nSample:")
    print(df[['date_str','delayed','dep_delay','wind_cmx','snow_cmx','wind_ord','snow_ord']].head(8).to_string(index=False))
