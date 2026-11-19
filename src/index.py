"""
Index construction and econometric preprocessing module for the
Emerging Markets Opportunity Dashboard.

Steps implemented:
1. Country completeness filter: drop countries missing >30% of scored indicators
   and export the dropped list for README transparency.
2. Imputation: fill isolated missing values (<30% per country) using the median
   of the country's income group (fallback: overall sample median).
3. Outlier handling: winsorize indicators at the 2.5th and 97.5th percentiles
   so extreme macro shocks (e.g., hyperinflation, FDI tax-haven spikes) do not
   compress the 0-100 scale for all other economies.
4. Min-max normalization (0-100): invert indicators where higher values mean
   worse conditions (inflation and external debt), so 100 always represents the
   most favorable condition (highest growth/stability, lowest inflation/debt).
5. Block & composite scoring: compute equal-weighted block scores (0-100) and
   user-weighted composite scores.
"""

from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "Growth": 0.30,
    "Demographics": 0.15,
    "Digitalization": 0.20,
    "Business Environment": 0.20,
    "Risk": 0.15,
}


BLOCK_INDICATORS = {
    "Growth": [
        {"code": "NY.GDP.PCAP.KD", "label": "GDP per capita (const. 2015 US$)", "invert": False},
        {"code": "NY.GDP.MKTP.KD.ZG", "label": "Latest GDP growth (%)", "invert": False},
        {"code": "GDP_GROWTH_5Y_AVG", "label": "5-Yr Avg GDP growth (%)", "invert": False},
    ],
    "Demographics": [
        {"code": "SP.POP.1564.TO.ZS", "label": "Working-age pop. 15-64 (%)", "invert": False},
        {"code": "SP.URB.TOTL.IN.ZS", "label": "Urban population (%)", "invert": False},
        {"code": "SP.POP.GROW", "label": "Population growth (%)", "invert": False},
    ],
    "Digitalization": [
        {"code": "IT.NET.USER.ZS", "label": "Internet users (% pop.)", "invert": False},
        {"code": "IT.CEL.SETS.P2", "label": "Mobile subscriptions (per 100)", "invert": False},
        {"code": "FX.OWN.TOTL.ZS", "label": "Financial account ownership (%)", "invert": False},
    ],
    "Business Environment": [
        {"code": "GE.EST", "label": "Government Effectiveness (WGI)", "invert": False},
        {"code": "RQ.EST", "label": "Regulatory Quality (WGI)", "invert": False},
        {"code": "BX.KLT.DINV.WD.GD.ZS", "label": "FDI net inflows (% GDP)", "invert": False},
    ],
    "Risk": [
        {"code": "FP.CPI.TOTL.ZG", "label": "Inflation, CPI (%)", "invert": True},
        {"code": "DT.DOD.DECT.GN.ZS", "label": "External debt (% GNI)", "invert": True},
        {"code": "BN.CAB.XOKA.GD.ZS", "label": "Current account balance (% GDP)", "invert": False},
        {"code": "PV.EST", "label": "Political Stability (WGI)", "invert": False},
    ],
}

MAX_COUNTRY_MISSING_SHARE = 0.30
WINSOR_LOWER_Q = 0.025
WINSOR_UPPER_Q = 0.975


def get_all_scored_codes() -> list[str]:
    """Return flat list of all 15 scored indicator codes across the 5 blocks."""
    return [item["code"] for items in BLOCK_INDICATORS.values() for item in items]


def filter_incomplete_countries(
    df: pd.DataFrame, max_missing: float = MAX_COUNTRY_MISSING_SHARE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop countries missing more than `max_missing` (30%) of scored indicators.

    Returns:
        (kept_df, dropped_df)
    """
    scored_cols = get_all_scored_codes()
    missing_share = df[scored_cols].isna().mean(axis=1)

    df = df.copy()
    df["missing_indicator_pct"] = (missing_share * 100).round(1)

    kept_df = df[missing_share <= max_missing].reset_index(drop=True)
    dropped_df = (
        df[missing_share > max_missing][["iso3", "country", "region", "income_group", "missing_indicator_pct"]]
        .sort_values("missing_indicator_pct", ascending=False)
        .reset_index(drop=True)
    )
    return kept_df, dropped_df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute isolated remaining NaNs using the median within the same World Bank
    income group (`income_group_code`), falling back to the full-sample median.
    """
    df = df.copy()
    for code in get_all_scored_codes():
        group_medians = df.groupby("income_group_code")[code].transform("median")
        df[code] = df[code].fillna(group_medians)
        df[code] = df[code].fillna(df[code].median())
    return df


def winsorize_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Winsorize each scored indicator at the 2.5th and 97.5th percentiles, then
    apply 0-100 min-max normalization. Indicators marked `invert=True` are
    reversed so 100 always represents the most favorable economic outcome.

    Adds normalized columns named `<code_name>_norm` to the DataFrame.
    """
    df = df.copy()
    for block_items in BLOCK_INDICATORS.values():
        for item in block_items:
            code = item["code"]
            invert = item["invert"]

            low = df[code].quantile(WINSOR_LOWER_Q)
            high = df[code].quantile(WINSOR_UPPER_Q)
            clipped = df[code].clip(lower=low, upper=high)

            denom = high - low
            if denom == 0 or np.isnan(denom):
                norm = pd.Series(50.0, index=df.index)
            else:
                norm = (clipped - low) / denom * 100.0

            if invert:
                norm = 100.0 - norm

            df[f"{code}_norm"] = norm.round(2)

    return df


def compute_composite_scores(
    df: pd.DataFrame, weights: dict[str, float] | None = None
) -> pd.DataFrame:
    """
    Compute equal-weighted block scores (0-100) and the composite Opportunity Score
    using normalized user weights.

    Also computes:
    - `opportunity_ex_risk`: Weighted average of the 4 non-risk blocks (0-100),
      used on the Y-axis of the Opportunity vs. Risk scatter plot so Risk is not
      double-counted across both axes.
    - `risk_exposure_score`: 100 - score_Risk (0 = lowest risk, 100 = highest risk),
      providing an intuitive X-axis for Opportunity vs. Risk visualization.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Normalize weights so they always sum to 1.0 (100%)
    total_w = sum(weights.values())
    if total_w <= 0:
        norm_w = {k: 1.0 / len(BLOCK_INDICATORS) for k in BLOCK_INDICATORS}
    else:
        norm_w = {k: v / total_w for k, v in weights.items()}

    df = df.copy()

    # 1. Block scores (simple average of normalized indicators inside each block)
    for block_name, items in BLOCK_INDICATORS.items():
        norm_cols = [f"{item['code']}_norm" for item in items]
        df[f"score_{block_name}"] = df[norm_cols].mean(axis=1).round(2)

    # 2. Composite score (weighted sum across all 5 blocks)
    composite = pd.Series(0.0, index=df.index)
    for block_name, w in norm_w.items():
        composite += df[f"score_{block_name}"] * w
    df["composite_score"] = composite.round(2)

    # 3. Separate Opportunity-only (4 blocks) vs. Risk Exposure 
    opp_blocks = ["Growth", "Demographics", "Digitalization", "Business Environment"]
    opp_w_sum = sum(norm_w[b] for b in opp_blocks)
    if opp_w_sum > 0:
        opp_ex_risk = sum(df[f"score_{b}"] * (norm_w[b] / opp_w_sum) for b in opp_blocks)
    else:
        opp_ex_risk = df[[f"score_{b}" for b in opp_blocks]].mean(axis=1)

    df["opportunity_ex_risk"] = opp_ex_risk.round(2)
    df["risk_exposure_score"] = (100.0 - df["score_Risk"]).round(2)

    # 4. Sort descending by composite score and assign 1-based rank
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def build_clean_dataset(raw_csv_path: Path, data_dir: Path) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline: load raw cross-section, drop >30% missing
    countries, impute isolated gaps, winsorize/normalize, compute default scores,
    and save `clean_em_index.csv` + `dropped_countries.csv`.
    """
    df_raw = pd.read_csv(raw_csv_path)
    df_kept, df_dropped = filter_incomplete_countries(df_raw)
    df_imputed = impute_missing_values(df_kept)
    df_norm = winsorize_and_normalize(df_imputed)
    df_scored = compute_composite_scores(df_norm, DEFAULT_WEIGHTS)

    clean_out = data_dir / "clean_em_index.csv"
    dropped_out = data_dir / "dropped_countries.csv"

    df_scored.to_csv(clean_out, index=False)
    df_dropped.to_csv(dropped_out, index=False)

    print(f"Kept countries:    {len(df_scored)} (saved to {clean_out})")
    print(f"Dropped countries: {len(df_dropped)} (saved to {dropped_out})")
    if not df_dropped.empty:
        print("\nDropped countries (>30% missing indicators):")
        print(df_dropped.to_string(index=False))

    print("\nTOP 15 EMERGING MARKETS (Default Weights):")
    display_cols = [
        "rank", "country", "region", "income_group_code",
        "composite_score", "score_Growth", "score_Demographics",
        "score_Digitalization", "score_Business Environment", "score_Risk",
    ]
    print(df_scored[display_cols].head(15).to_string(index=False))
    return df_scored


if __name__ == "__main__":
    base_data_dir = Path(__file__).resolve().parent.parent / "data"
    build_clean_dataset(base_data_dir / "raw_em_data.csv", base_data_dir)