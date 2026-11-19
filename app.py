import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Emerging Markets Opportunity Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

PILLARS = {
    "Growth": "score_Growth",
    "Demographics": "score_Demographics",
    "Digitalization": "score_Digitalization",
    "Business Environment": "score_Business Environment",
    "Risk": "score_Risk",
}

DEFAULT_WEIGHTS = {
    "Growth": 30,
    "Demographics": 15,
    "Digitalization": 20,
    "Business Environment": 20,
    "Risk": 15,
}


@st.cache_data
def load_data():
    df_index = pd.read_csv("data/clean_em_index.csv")
    df_panel = pd.read_csv("data/panel_em_data.csv")
    df_sens = pd.read_csv("data/sensitivity_summary.csv")
    df_audit = pd.read_csv("data/indicator_coverage_audit.csv")
    df_dropped = pd.read_csv("data/dropped_countries.csv")
    return df_index, df_panel, df_sens, df_audit, df_dropped


df_base, df_panel, df_sens, df_audit, df_dropped = load_data()
series_label_map = dict(zip(df_audit["code"], df_audit["name"]))

# --- SIDEBAR: DYNAMIC WEIGHTS & FILTERS ---
st.sidebar.header("Index Weights")
st.sidebar.caption("Adjust block priorities below. Weights are automatically normalized to 100%.")

raw_weights = {}
for label, default_val in DEFAULT_WEIGHTS.items():
    raw_weights[label] = st.sidebar.slider(label, min_value=0, max_value=100, value=default_val, step=5)

total_raw = sum(raw_weights.values())
if total_raw == 0:
    weights = {k: 0.20 for k in DEFAULT_WEIGHTS}
else:
    weights = {k: v / total_raw for k, v in raw_weights.items()}

with st.sidebar.expander("Effective Normalized Weights", expanded=False):
    for label, w in weights.items():
        st.write(f"**{label}:** {w * 100:.1f}%")

st.sidebar.divider()
st.sidebar.header("Market Filters")
all_regions = sorted(df_base["region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect("World Bank Region", options=all_regions, default=all_regions)

all_incomes = sorted(df_base["income_group"].dropna().unique().tolist())
selected_incomes = st.sidebar.multiselect("Income Group", options=all_incomes, default=all_incomes)

# --- RECOMPUTE COMPOSITE SCORE IN REAL TIME ---
df = df_base.copy()
df["composite_score"] = sum(df[col] * weights[label] for label, col in PILLARS.items())
df["global_rank"] = df["composite_score"].rank(ascending=False, method="min").astype(int)
df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

# Opportunity score excluding Risk (prevents double-counting on 2D scatter)
opp_pillars = ["Growth", "Demographics", "Digitalization", "Business Environment"]
opp_weight_sum = sum(weights[p] for p in opp_pillars)
if opp_weight_sum > 0:
    df["opportunity_score"] = sum(df[PILLARS[p]] * (weights[p] / opp_weight_sum) for p in opp_pillars)
else:
    df["opportunity_score"] = df[[PILLARS[p] for p in opp_pillars]].mean(axis=1)

df["risk_exposure"] = 100.0 - df["score_Risk"]

df_filtered = df[df["region"].isin(selected_regions) & df["income_group"].isin(selected_incomes)].copy()

# --- HEADER & KPI CARDS ---
st.title("Emerging Markets Opportunity Dashboard")
st.markdown(
    "**Guiding Question:** *Which emerging markets offer the most growth potential, and what are their risks?*  \n"
    f"Interactive composite index across **{len(df)} emerging & developing economies** "
    "(population ≥ 1M; World Bank LIC, LMC, and UMC groups)."
)

if df_filtered.empty:
    st.warning("No economies match your current Region / Income Group filters. Adjust the sidebar filters to view results.")
    st.stop()

top_row = df_filtered.iloc[0]
top_pillar = max(weights, key=weights.get)

k1, k2, k3, k4 = st.columns(4)
k1.metric("#1 Ranked Market (Filtered)", top_row["country"], f"Score: {top_row['composite_score']:.1f} / 100")
k2.metric("Markets Displayed", f"{len(df_filtered)} / {len(df)}", f"{len(selected_regions)} regions active")
k3.metric("Median Composite Score", f"{df_filtered['composite_score'].median():.1f}", f"Max: {df_filtered['composite_score'].max():.1f}")
k4.metric("Top Weight Priority", top_pillar, f"{weights[top_pillar] * 100:.0f}% weight")

st.divider()

# --- 4 ANALYTICAL TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Global Rankings & Map",
    "2. Opportunity vs. Risk & Robustness",
    "3. Country Deep-Dive",
    "4. Compare & Methodology",
])

# === TAB 1: GLOBAL RANKINGS & MAP ===
with tab1:
    st.subheader("Global Opportunity Heatmap")
    fig_map = px.choropleth(
        df_filtered,
        locations="iso3",
        color="composite_score",
        hover_name="country",
        hover_data={
            "iso3": False,
            "global_rank": True,
            "composite_score": ":.1f",
            "score_Growth": ":.1f",
            "score_Risk": ":.1f",
            "region": True,
        },
        color_continuous_scale="Tealgrn",
        range_color=[20, 85],
        labels={
            "composite_score": "Score (0–100)",
            "global_rank": "Global Rank",
            "score_Growth": "Growth",
            "score_Risk": "Risk Safety",
            "region": "Region",
        },
    )
    fig_map.update_geos(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#444",
        projection_type="natural earth",
        bgcolor="rgba(0,0,0,0)",
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=440,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Ranked Market Scorecard")
    st.caption("All block scores are normalized from 0 to 100 (higher = stronger fundamentals / lower risk).")

    display_cols = [
        "global_rank", "country", "region", "income_group", "composite_score",
        "score_Growth", "score_Demographics", "score_Digitalization",
        "score_Business Environment", "score_Risk"
    ]
    rename_map = {
        "global_rank": "Global Rank",
        "country": "Country",
        "region": "Region",
        "income_group": "Income",
        "composite_score": "Composite Score",
        "score_Growth": f"Growth ({weights['Growth']*100:.0f}%)",
        "score_Demographics": f"Demog. ({weights['Demographics']*100:.0f}%)",
        "score_Digitalization": f"Digital ({weights['Digitalization']*100:.0f}%)",
        "score_Business Environment": f"Business ({weights['Business Environment']*100:.0f}%)",
        "score_Risk": f"Risk Safety ({weights['Risk']*100:.0f}%)",
    }
    table_df = df_filtered[display_cols].rename(columns=rename_map)

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Composite Score": st.column_config.ProgressColumn(
                "Composite Score", format="%.1f", min_value=0, max_value=100
            ),
            rename_map["score_Growth"]: st.column_config.NumberColumn(format="%.1f"),
            rename_map["score_Demographics"]: st.column_config.NumberColumn(format="%.1f"),
            rename_map["score_Digitalization"]: st.column_config.NumberColumn(format="%.1f"),
            rename_map["score_Business Environment"]: st.column_config.NumberColumn(format="%.1f"),
            rename_map["score_Risk"]: st.column_config.NumberColumn(format="%.1f"),
        },
    )

    csv_export = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Rankings (CSV)",
        data=csv_export,
        file_name="em_opportunity_rankings.csv",
        mime="text/csv",
    )

# === TAB 2: OPPORTUNITY VS RISK & ROBUSTNESS ===
with tab2:
    st.subheader("Opportunity vs. Risk Exposure Matrix")
    st.caption(
        "Opportunity Score averages the 4 structural upside pillars (excluding Risk to avoid double-counting). "
        "Moving right on the X-axis indicates higher macroeconomic & political risk exposure (100 − Risk Safety)."
    )

    med_opp = df_filtered["opportunity_score"].median()
    med_risk = df_filtered["risk_exposure"].median()

    fig_scatter = px.scatter(
        df_filtered,
        x="risk_exposure",
        y="opportunity_score",
        color="region",
        size="composite_score",
        hover_name="country",
        text=df_filtered.apply(lambda r: r["country"] if r["global_rank"] <= 12 else "", axis=1),
        hover_data={
            "global_rank": True,
            "composite_score": ":.1f",
            "opportunity_score": ":.1f",
            "risk_exposure": ":.1f",
            "region": False,
        },
        labels={
            "risk_exposure": "Risk Exposure (100 − Risk Safety Score → Higher = Riskier)",
            "opportunity_score": "Structural Opportunity Score (0–100)",
            "region": "Region",
        },
    )
    fig_scatter.update_traces(textposition="top center", textfont_size=10)
    fig_scatter.add_vline(x=med_risk, line_dash="dash", line_color="#666")
    fig_scatter.add_hline(y=med_opp, line_dash="dash", line_color="#666")
    fig_scatter.update_layout(height=500, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    st.subheader("Monte Carlo Weight-Sensitivity Robustness (500 Simulations)")
    st.caption(
        "Top-10 rank stability under 500 random Dirichlet weight perturbations around the baseline index."
    )
    top_sens = df_sens.sort_values("top10_prob_pct", ascending=False).head(15)
    fig_sens = px.bar(
        top_sens,
        x="top10_prob_pct",
        y="country",
        orientation="h",
        color="mean_sim_rank",
        color_continuous_scale="Tealgrn_r",
        labels={
            "top10_prob_pct": "% of Simulations Ranked in Top 10",
            "country": "Economy",
            "mean_sim_rank": "Mean Rank",
        },
    )
    fig_sens.update_layout(yaxis=dict(autorange="reversed"), height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_sens, use_container_width=True)

# === TAB 3: COUNTRY DEEP-DIVE ===
with tab3:
    selected_country = st.selectbox(
        "Select an Emerging Market to Inspect",
        options=df["country"].tolist(),
        index=0,
    )
    c_row = df[df["country"] == selected_country].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global Rank", f"#{int(c_row['global_rank'])} / {len(df)}", c_row["region"])
    c2.metric("Composite Score", f"{c_row['composite_score']:.1f} / 100", f"Income: {c_row['income_group']}")
    c3.metric("Structural Opportunity", f"{c_row['opportunity_score']:.1f} / 100", "4 upside pillars")
    c4.metric("Risk Safety Score", f"{c_row['score_Risk']:.1f} / 100", "Higher = safer")

    col_radar, col_notes = st.columns([1.2, 1])

    with col_radar:
        pillar_names = list(PILLARS.keys())
        country_vals = [c_row[PILLARS[p]] for p in pillar_names]
        median_vals = [df[PILLARS[p]].median() for p in pillar_names]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=country_vals + [country_vals[0]],
            theta=pillar_names + [pillar_names[0]],
            fill="toself",
            name=selected_country,
            line_color="#00CC96",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=median_vals + [median_vals[0]],
            theta=pillar_names + [pillar_names[0]],
            fill="toself",
            name="EM Median (104 Markets)",
            line_color="#888888",
            opacity=0.5,
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=400,
            margin=dict(l=40, r=40, t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_notes:
        st.subheader("Automated Diagnostic")
        pillar_scores = {p: c_row[PILLARS[p]] for p in pillar_names}
        sorted_pillars = sorted(pillar_scores.items(), key=lambda x: x[1], reverse=True)

        st.markdown("**Key Strengths:**")
        for p_name, p_val in sorted_pillars[:2]:
            diff = p_val - df[PILLARS[p_name]].median()
            st.write(f"- **{p_name}:** `{p_val:.1f} / 100` ({diff:+.1f} vs. EM median)")

        st.markdown("**Primary Vulnerabilities:**")
        for p_name, p_val in sorted_pillars[-2:]:
            diff = p_val - df[PILLARS[p_name]].median()
            st.write(f"- **{p_name}:** `{p_val:.1f} / 100` ({diff:+.1f} vs. EM median)")

    # Historical Trajectory (2013-2025 wide panel)
    c_panel = df_panel[df_panel["economy"] == c_row["iso3"]].copy()
    if not c_panel.empty:
        st.subheader(f"Historical Macro Trajectory — {selected_country}")
        available_series = c_panel["series"].dropna().unique().tolist()
        if available_series:
            chosen_series = st.selectbox(
                "Select Historical Indicator",
                options=available_series,
                format_func=lambda s: f"{series_label_map.get(s, s)} ({s})",
            )
            yr_cols = [col for col in c_panel.columns if col.startswith("YR")]
            s_row = c_panel[c_panel["series"] == chosen_series].iloc[0]
            hist_df = pd.DataFrame({
                "Year": [int(y.replace("YR", "")) for y in yr_cols],
                "Value": [s_row[y] for y in yr_cols],
            }).dropna()

            fig_hist = px.line(
                hist_df,
                x="Year",
                y="Value",
                markers=True,
                title=f"{selected_country}: {series_label_map.get(chosen_series, chosen_series)}",
            )
            fig_hist.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_hist, use_container_width=True)

# === TAB 4: COMPARE & METHODOLOGY ===
with tab4:
    st.subheader("Side-by-Side Peer Comparison")
    default_peers = [c for c in ["Malaysia", "India", "Brazil"] if c in df["country"].values]
    if len(default_peers) < 2:
        default_peers = df["country"].head(3).tolist()

    peers = st.multiselect(
        "Select 2 to 4 Economies to Compare",
        options=df["country"].tolist(),
        default=default_peers,
        max_selections=4,
    )

    if len(peers) >= 2:
        peer_df = df[df["country"].isin(peers)]
        fig_comp = go.Figure()
        pillar_names = list(PILLARS.keys())
        for _, p_row in peer_df.iterrows():
            vals = [p_row[PILLARS[p]] for p in pillar_names]
            fig_comp.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=pillar_names + [pillar_names[0]],
                fill="toself",
                name=f"{p_row['country']} (#{int(p_row['global_rank'])})",
            ))
        fig_comp.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=430,
            margin=dict(l=40, r=40, t=30, b=30),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        comp_cols = ["global_rank", "country", "region", "income_group", "composite_score"] + list(PILLARS.values())
        st.dataframe(peer_df[comp_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Data Coverage Audit & Dropped Economies")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Indicator Coverage Audit**")
        st.dataframe(df_audit, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Excluded Economies (>30% Missingness)**")
        st.dataframe(df_dropped, use_container_width=True, hide_index=True)