"""
Data acquisition and audit pipeline for the Emerging Markets Opportunity Dashboard.

Pulls World Development Indicators (WDI) and Worldwide Governance Indicators (WGI)
from the official World Bank API (`wbgapi` metadata + bulk CSV endpoint with JSON
API fallback), filters to emerging and developing economies with population >= 1M,
computes multi-year growth averages, extracts the most recent value within a
5-year window, and outputs a coverage audit alongside `data/raw_em_data.csv`.
"""

from datetime import datetime, timezone
import io
from pathlib import Path
import time
import zipfile
import numpy as np
import pandas as pd
import requests
import wbgapi as wb


INDICATORS = {
    # Population filter )
    "SP.POP.TOTL": "Population, total",

    # Block 1: Growth
    "NY.GDP.PCAP.KD": "GDP per capita (constant 2015 US$)",
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",

    # Block 2: Demographics
    "SP.POP.1564.TO.ZS": "Working-age population (% of total, ages 15-64)",
    "SP.URB.TOTL.IN.ZS": "Urban population (% of total)",
    "SP.POP.GROW": "Population growth (annual %)",

    # Block 3: Digitalization
    "IT.NET.USER.ZS": "Internet users (% of population)",
    "IT.CEL.SETS.P2": "Mobile cellular subscriptions (per 100 people)",
    "FX.OWN.TOTL.ZS": "Account ownership at financial institution/mobile (% age 15+)",

    # Block 4: Business Environment
    "GE.EST": "Government Effectiveness (WGI estimate, -2.5 to +2.5)",
    "RQ.EST": "Regulatory Quality (WGI estimate, -2.5 to +2.5)",
    "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment, net inflows (% of GDP)",

    # Block 5: Risk
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "DT.DOD.DECT.GN.ZS": "External debt stocks (% of GNI)",
    "BN.CAB.XOKA.GD.ZS": "Current account balance (% of GDP)",
    "PV.EST": "Political Stability & Absence of Violence (WGI estimate)",
}

TARGET_INCOME_GROUPS = {"LIC", "LMC", "UMC"}
MIN_POPULATION = 1_000_000
LOOKBACK_YEARS = 12       
RECENT_WINDOW_YEARS = 5   


def get_eligible_economies() -> pd.DataFrame:
    """
    Fetch country metadata from the World Bank via `wbgapi` and keep only sovereign
    emerging/developing countries (LIC, LMC, UMC), excluding aggregates.
    """
    print("1/4 Fetching World Bank economy metadata...")

    income_labels = {
        "LIC": "Low income",
        "LMC": "Lower middle income",
        "UMC": "Upper middle income",
    }
    region_labels = wb.region.Series().to_dict()

    eco_df = wb.economy.DataFrame()
    eco_df = eco_df[~eco_df["aggregate"]].copy()
    eco_df = eco_df[eco_df["incomeLevel"].isin(TARGET_INCOME_GROUPS)].copy()

    df_meta = pd.DataFrame({
        "iso3": eco_df.index,
        "country": eco_df["name"].values,
        "region": eco_df["region"].map(region_labels).fillna(eco_df["region"]).values,
        "income_group_code": eco_df["incomeLevel"].values,
        "income_group": eco_df["incomeLevel"].map(income_labels).values,
    }).sort_values("country").reset_index(drop=True)

    print(f"    -> Found {len(df_meta)} LIC/LMC/UMC countries before population filter.")
    return df_meta


def fetch_via_json_fallback(
    code: str, start_year: int, end_year: int, valid_iso3: set[str]
) -> pd.DataFrame:
    """
    Fallback fetcher using the paginated JSON endpoint when the bulk CSV ZIP times out.
    """
    url = (
        f"https://api.worldbank.org/v2/country/all/indicator/{code}"
        f"?date={start_year}:{end_year}&format=json&per_page=10000"
    )
    headers = {"User-Agent": "Mozilla/5.0 (EM-Opportunity-Dashboard/1.0)"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise ValueError(f"Indicator '{code}' not found in JSON fallback.")

    records = [
        {
            "economy": item["countryiso3code"],
            "year": int(item["date"]),
            "value": item["value"],
        }
        for item in payload[1]
        if item.get("countryiso3code") in valid_iso3
    ]
    df_long = pd.DataFrame(records).drop_duplicates(subset=["economy", "year"])
    df_wide = df_long.pivot(index="economy", columns="year", values="value")
    df_wide.columns = [f"YR{y}" for y in df_wide.columns]
    df_wide["series"] = code
    return df_wide.reset_index().set_index(["economy", "series"])


def fetch_single_indicator(
    code: str, start_year: int, end_year: int, valid_iso3: set[str]
) -> pd.DataFrame:
    """
    Download an indicator via World Bank's fast bulk CSV ZIP endpoint, automatically
    falling back to the JSON API if the CSV generator times out.
    """
    url = f"https://api.worldbank.org/v2/en/indicator/{code}?downloadformat=csv"
    headers = {"User-Agent": "Mozilla/5.0 (EM-Opportunity-Dashboard/1.0)"}

    content = None
    for attempt in range(1, 3):
        try:
            resp = requests.get(url, headers=headers, timeout=45)
            resp.raise_for_status()
            content = resp.content
            break
        except requests.exceptions.RequestException:
            if attempt == 1:
                print("(retrying CSV)...", end=" ", flush=True)
                time.sleep(2)

    if content is None:
        print("(switching to JSON API)...", end=" ", flush=True)
        return fetch_via_json_fallback(code, start_year, end_year, valid_iso3)

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        data_files = [name for name in zf.namelist() if name.startswith("API_") and name.endswith(".csv")]
        if not data_files:
            raise ValueError(f"Indicator '{code}' not found in World Bank CSV archive.")
        with zf.open(data_files[0]) as f:
            df_raw = pd.read_csv(f, skiprows=4)

    df_raw = df_raw[df_raw["Country Code"].isin(valid_iso3)].copy()
    df_raw = df_raw.drop_duplicates(subset=["Country Code"]).copy()
    if df_raw.empty:
        raise ValueError(f"No target countries matched for '{code}'.")

    year_cols = [str(y) for y in range(start_year, end_year + 1) if str(y) in df_raw.columns]
    df_sub = df_raw[["Country Code"] + year_cols].copy()
    df_sub = df_sub.rename(columns={"Country Code": "economy"})
    df_sub = df_sub.rename(columns={y: f"YR{y}" for y in year_cols})

    yr_prefixed = [f"YR{y}" for y in year_cols]
    for col in yr_prefixed:
        df_sub[col] = pd.to_numeric(df_sub[col], errors="coerce")

    df_sub["series"] = code
    return df_sub.set_index(["economy", "series"])


def download_panel_data(iso3_list: list[str], indicator_codes: list[str]) -> pd.DataFrame:
    """
    Download historical annual panel data (2013-2025) indicator-by-indicator.
    """
    end_year = datetime.now().year - 1  # 2025
    start_year = end_year - LOOKBACK_YEARS
    valid_iso3 = set(iso3_list)

    print(f"2/4 Downloading panel data ({start_year}-{end_year}) for {len(indicator_codes)} series...")
    frames = []
    for idx, code in enumerate(indicator_codes, start=1):
        print(f"    [{idx:02d}/{len(indicator_codes):02d}] Fetching {code}...", end=" ", flush=True)
        try:
            df_ind = fetch_single_indicator(code, start_year, end_year, valid_iso3)
            frames.append(df_ind)
            print("OK")
        except Exception as exc:
            print(f"FAILED ({exc})")

    if not frames:
        raise RuntimeError("No indicators could be downloaded. Check internet connection.")

    combined = pd.concat(frames)
    
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def extract_cross_section(raw_df: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the multi-year panel into a single cross-sectional row per country:
    - Takes the latest non-null value within the last `RECENT_WINDOW_YEARS` (2021-2025).
    - Computes a 5-year average annual GDP growth rate (`GDP_GROWTH_5Y_AVG`) to smooth
      single-year post-pandemic or commodity shocks.
    """
    print("3/4 Extracting latest available values (2021-2025) and 5Y average GDP growth...")

    year_cols = sorted([c for c in raw_df.columns if c.startswith("YR")])
    year_ints = [int(c.replace("YR", "")) for c in year_cols]
    max_year = max(year_ints)
    cutoff_year = max_year - RECENT_WINDOW_YEARS + 1
    recent_cols = [f"YR{y}" for y in year_ints if y >= cutoff_year]

    
    latest_series = raw_df[recent_cols].ffill(axis=1).iloc[:, -1]
    cross_df = latest_series.unstack(level="series").reset_index()
    cross_df = cross_df.rename(columns={"economy": "iso3"})

    
    gdp_growth_panel = raw_df.xs("NY.GDP.MKTP.KD.ZG", level="series")[year_cols]

    def trailing_5y_mean(row: pd.Series) -> float:
        valid = row.dropna()
        if len(valid) < 3:
            return np.nan
        return float(valid.iloc[-5:].mean())

    avg_growth = gdp_growth_panel.apply(trailing_5y_mean, axis=1).rename("GDP_GROWTH_5Y_AVG")
    cross_df = cross_df.merge(
        avg_growth.reset_index().rename(columns={"economy": "iso3"}),
        on="iso3",
        how="left",
    )

    
    merged = df_meta.merge(cross_df, on="iso3", how="left")
    before_pop = len(merged)
    merged = merged[merged["SP.POP.TOTL"] >= MIN_POPULATION].copy()
    print(f"    -> Excluded {before_pop - len(merged)} countries with population < 1M (or missing pop).")
    print(f"    -> Remaining sample: {len(merged)} countries.")

    return merged.reset_index(drop=True)


def run_coverage_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify every indicator exists and report % missing across the eligible country sample.
    Flags any indicator exceeding the 30% missingness threshold.
    """
    print("\n4/4 INDICATOR COVERAGE AUDIT (Rule: <= 30% missing)")
    print("-" * 78)
    print(f"{'Code':<22} | {'Missing':>8} | {'Status':<10} | {'Indicator Name'}")
    print("-" * 78)

    scored_cols = [c for c in INDICATORS.keys() if c != "SP.POP.TOTL"] + ["GDP_GROWTH_5Y_AVG"]
    audit_rows = []

    for code in scored_cols:
        name = INDICATORS.get(code, "GDP growth, 5-year trailing average (annual %)")
        if code not in df.columns:
            pct_missing = 100.0
        else:
            pct_missing = df[code].isna().mean() * 100

        status = "PASS" if pct_missing <= 30.0 else "FAIL (>30%)"
        print(f"{code:<22} | {pct_missing:7.1f}% | {status:<10} | {name[:36]}")
        audit_rows.append({
            "code": code,
            "name": name,
            "pct_missing": round(pct_missing, 2),
            "status": status,
        })

    print("-" * 78)
    return pd.DataFrame(audit_rows)


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df_meta = get_eligible_economies()
    raw_panel = download_panel_data(df_meta["iso3"].tolist(), list(INDICATORS.keys()))
    df_clean_raw = extract_cross_section(raw_panel, df_meta)

    df_clean_raw["data_fetched_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    audit_df = run_coverage_audit(df_clean_raw)

    raw_out_path = data_dir / "raw_em_data.csv"
    audit_out_path = data_dir / "indicator_coverage_audit.csv"

    df_clean_raw.to_csv(raw_out_path, index=False)
    audit_df.to_csv(audit_out_path, index=False)
    panel_out_path = data_dir / "panel_em_data.csv"
    raw_panel.reset_index().to_csv(panel_out_path, index=False)
    print(f"Saved historical panel to:  {panel_out_path}")
    print(f"\nSaved raw cross-section to: {raw_out_path}")
    print(f"Saved coverage audit to:    {audit_out_path}")


if __name__ == "__main__":
    main()