# Emerging Markets Opportunity Dashboard

**Live Interactive Dashboard:** https://em-opportunity-dashboard-fhhog9qkgtmtpylxptpnse.streamlit.app/

An interactive macroeconomic and institutional decision-support dashboard evaluating **104 emerging and developing economies** across five structural pillars: **Growth**, **Demographics**, **Digitalization**, **Business Environment**, and **Macroeconomic & Political Risk**.

> **Guiding Question:** *Which emerging markets offer the strongest long-term growth potential, and what are their underlying macroeconomic and institutional risks?*

---

## Key Features
- **Dynamic Composite Index:** Real-time weight sliders across 5 pillars (default: 30% Growth, 15% Demographics, 20% Digitalization, 20% Business Environment, 15% Risk), auto-normalized to 100%.
- **Global Choropleth Heatmap & Ranked Scorecard:** Interactive World Bank region and income-group filtering with downloadable CSV rankings.
- **Opportunity vs. Risk Matrix & Monte Carlo Robustness:** 2D quadrant positioning (excluding Risk from the Opportunity axis to avoid double-counting) paired with 500 Dirichlet weight-perturbation simulations measuring Top-10 rank stability.
- **Country Deep-Dive & Peer Comparison:** 5-pillar radar profiles against the emerging-market median, automated top strengths/vulnerabilities, 10-year historical trajectories (2015–2025), and 2–3 country side-by-side comparisons.

---

## Methodology & Data Pipeline
1. **Sample Selection (`src/data.py`):** Sovereign World Bank *Low Income (LIC)*, *Lower Middle Income (LMC)*, and *Upper Middle Income (UMC)* economies with population $\ge 1,000,000$. Countries missing $>30\%$ of scored indicators (6 states: Eritrea, North Korea, Yemen, Cuba, South Sudan, Syria) are excluded and logged in `data/dropped_countries.csv`, leaving **104 economies**.
2. **Preprocessing & Winsorization (`src/index.py`):** Isolated missing observations are imputed via World Bank income-group medians. All 15 sub-indicators are winsorized at the **2.5th and 97.5th percentiles** so extreme hyperinflation or FDI conduit spikes do not compress the $[0, 100]$ scale.
3. **Direction-Consistent Normalization:** Min-max scaled to $[0, 100]$, inverting **CPI Inflation** and **External Debt (% of GNI)** so `100` uniformly represents the strongest fundamental / lowest risk.

### Empirical Validation (`notebooks/analysis.ipynb`)
- **Block Independence:** Inter-pillar Spearman rank correlations range from $\rho = 0.14$ to $0.53$, confirming low redundancy across the 5 blocks.
- **IMF WEO Alignment:** The Growth block correlates positively with **IMF WEO 2026–2028 projected real GDP growth** across $n = 99$ matched economies ($\rho = 0.42, p < 0.0001$).
- **Sovereign Credit Risk Alignment:** The Risk safety block correlates positively with numeric **S&P/Fitch sovereign credit ratings** across 32 major emerging markets ($\rho = 0.53$).

---

## Project Structure
```text
em-opportunity-dashboard/
├── app.py                  # Interactive 4-tab Streamlit application
├── requirements.txt        # Production dependencies for deployment
├── src/
│   ├── data.py             # World Bank WDI/WGI acquisition & coverage audit
│   └── index.py            # Cleaning, winsorization, normalization & scoring engine
├── notebooks/
│   └── analysis.ipynb      # EDA, IMF/credit rating validation & Monte Carlo sensitivity
└── data/
    ├── raw_em_data.csv
    ├── clean_em_index.csv
    ├── panel_em_data.csv
    ├── sensitivity_summary.csv
    ├── indicator_coverage_audit.csv
    └── dropped_countries.csv
```

---

## Run Locally
```bash
git clone https://github.com/ocereztr/em-opportunity-dashboard.git
cd em-opportunity-dashboard
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

*Data Sources: World Bank Open Data (WDI & WGI) and IMF World Economic Outlook (WEO). Built for academic and portfolio research purposes*