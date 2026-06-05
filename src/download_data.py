import os
import requests
import pandas as pd
from pathlib import Path
import time

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Catastrophic Health Expenditure — Final Data Download")
print("=" * 60)

# 1. World Bank (with retry)
print("\n[1/3] World Bank Indicators...")

WB_INDICATORS = {
    "SH.XPD.OOPC.CH.ZS": "oop_pct_current_health_exp",
    "SH.XPD.CHEX.GD.ZS": "health_exp_pct_gdp",
    "SH.XPD.PVTD.CH.ZS": "private_health_exp_pct",
    "SH.XPD.GHED.GD.ZS": "govt_health_exp_pct_gdp",   # government health % GDP
    "SP.POP.TOTL":       "total_population",
    "NY.GDP.PCAP.CD":    "gdp_per_capita_usd",
    "SI.POV.NAHC":       "poverty_headcount_pct",
}

frames = []
for code, name in WB_INDICATORS.items():
    for attempt in range(3):  # retry 3 times
        try:
            url = f"https://api.worldbank.org/v2/country/PK/indicator/{code}?format=json&per_page=100&mrv=30"
            r = requests.get(url, timeout=30)   # increased timeout
            r.raise_for_status()
            payload = r.json()
            
            if len(payload) > 1 and payload[1]:
                df = pd.DataFrame(payload[1])[["date", "value"]].rename(
                    columns={"date": "year", "value": name}
                )
                df["year"] = pd.to_numeric(df["year"])
                df = df.dropna(subset=[name])
                frames.append(df.set_index("year"))
                print(f"   ✓ {name}")
                break
            else:
                print(f"   ✗ {name} — no data")
                break
        except Exception as e:
            if attempt < 2:
                print(f"   ⚠ {name} — timeout, retrying... ({attempt+1}/3)")
                time.sleep(2)
            else:
                print(f"   ✗ {name} — failed after retries: {e}")

if frames:
    wb = frames[0]
    for f in frames[1:]:
        wb = wb.join(f, how="outer")
    wb = wb.sort_index(ascending=False).reset_index()
    wb.to_csv(RAW / "wb_pakistan_health_indicators.csv", index=False)
    print(f"\n   Saved: wb_pakistan_health_indicators.csv ({wb.shape})")

# 2. Kaggle
print("\n[2/3] Kaggle Healthcare Dataset...")
print("   ✓ Already exists (healthcare_dataset.csv)")

# 3. WHO
print("\n[3/3] WHO Catastrophic Expenditure...")
WHO_INDICATORS = {
    "FINPROTECTION_CATA_TOT_10_POP": "catastrophic_10pct",
    "FINPROTECTION_CATA_TOT_25_POP": "catastrophic_25pct",
}

who_frames = []
for code, name in WHO_INDICATORS.items():
    try:
        url = f"https://ghoapi.azureedge.net/api/{code}?$filter=SpatialDim eq 'PAK'"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json().get("value", [])
        if data:
            df = pd.DataFrame(data)[["TimeDim", "NumericValue"]].rename(
                columns={"TimeDim": "year", "NumericValue": name}
            )
            df = df.dropna(subset=[name])
            who_frames.append(df.set_index("year"))
            print(f"   ✓ {name}")
    except Exception as e:
        print(f"   ✗ {name} — {e}")

if who_frames:
    who = who_frames[0]
    for f in who_frames[1:]:
        who = who.join(f, how="outer")
    who = who.reset_index()
    who.to_csv(RAW / "who_pakistan_catastrophic_exp.csv", index=False)
    print(f"   Saved: who_pakistan_catastrophic_exp.csv")

# Final Summary
print("\n" + "="*70)
print("✅ Data Download Summary:")
for f in sorted(RAW.glob("*.csv")):
    size_mb = f.stat().st_size / (1024*1024)
    print(f"   {f.name:<45} {size_mb:>6.2f} MB")
print("="*70)
print("\nNext Step → Cleaning + EDA (Hefte 2)")