"""
TVS Credit EPIC 8 — Dynamic Residual Pricing & Lending Strategy Engine
Interactive app: (1) Portfolio Dashboard  (2) AI Lending Copilot  (3) Scenario Simulator.
Every number is produced by engine.py — the same validated pipeline used in the notebooks.
Run:  streamlit run app.py
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import streamlit as st

BASE = Path(__file__).parent; sys.path.insert(0, str(BASE))
import engine
OUT = BASE/"outputs"; BRAND = "#00447C"; RED = "#c0392b"

st.set_page_config(page_title="TVS Residual Pricing Engine", page_icon="🏍️", layout="wide")

@st.cache_resource(show_spinner="Loading models & scoring the portfolio…")
def load():
    df = pd.read_parquet(OUT/"scored_dataset.parquet")
    summ = json.loads((OUT/"optimization_summary.json").read_text())
    engine.load_assets()
    RV = engine.build_rv_matrix(df)
    return df, summ, RV
df, SUMM, RV = load()
H = SUMM["headline"]

def opts(col):  # dropdown choices from the data
    return sorted(df[col].dropna().astype(str).unique().tolist())

# ---------------------------------------------------------------- header
st.markdown(f"<h1 style='color:{BRAND};margin-bottom:0'>Dynamic Residual Pricing & Lending Strategy Engine</h1>"
            "<p style='color:gray;margin-top:2px'>TVS Credit · EPIC 8 Analytics Challenge · two-wheeler portfolio</p>",
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊  Portfolio Dashboard", "🤝  AI Lending Copilot", "🌪️  Scenario Simulator"])

# ================================================================ TAB 1 — DASHBOARD
with tab1:
    st.subheader("Portfolio impact of the recommended lending policy")
    c = st.columns(4)
    c[0].metric("Expected loss (current → recommended)",
                f"₹{H['gd_loss_recommended_cr']:.1f} Cr", f"{H['gd_loss_reduction_pct']:.0f}% vs ₹{H['gd_loss_current_cr']:.1f} Cr")
    c[1].metric("Mean LGD (severity)", f"{H['mean_lgd_recommended']*100:.1f}%",
                f"{(H['mean_lgd_recommended']-H['mean_lgd_current'])*100:.1f} pp")
    c[2].metric("Negative-equity window", f"{H['neg_equity_months_recommended']:.1f} mo",
                f"{H['neg_equity_months_recommended']-H['neg_equity_months_current']:.1f} mo")
    c[3].metric("Risk-adjusted profit", f"₹{H['profit_lift_cr']:+.2f} Cr", "break-even")

    st.caption("All models are **conditional on default** (the data is a repossession-only book): the engine "
               "optimises loss-given-default & recovery, taking probability-of-default as an input band.")

    a, b = st.columns(2)
    with a:
        st.markdown("**Residual-risk band distribution**")
        st.bar_chart(df["Risk_Band"].value_counts().reindex(["Low","Medium","High","Critical"]), color=BRAND)
    with b:
        st.markdown("**Segment risk — mean risk score by model**")
        seg = df.groupby("Asset Model")["Residual_Risk_Score"].mean().sort_values(ascending=False)
        st.bar_chart(seg, color=RED, horizontal=True)

    st.markdown("**LTV tightens monotonically with risk (current vs recommended)**")
    bb = df.groupby("Risk_Band", observed=True).agg(Current=("LTV","mean"), Recommended=("Rec_LTV","mean")
             ).reindex(["Low","Medium","High","Critical"])
    st.bar_chart(bb, color=[RED, BRAND], stack=False)

    with st.expander("Browse loan-level recommendations"):
        show = ["Agmt Id","Asset Model","Risk_Band","Residual_Risk_Score","LTV","Rec_LTV",
                "Cust Net IRR","Rec_Rate","Tenure","Rec_Tenure","Cur_E_LGD","Rec_E_LGD","NetValue_Lift"]
        st.dataframe(df[show].head(400), width="stretch", height=320)

# ================================================================ TAB 2 — COPILOT
with tab2:
    st.subheader("Price a loan & explain the decision")
    with st.form("loan"):
        c = st.columns(4)
        model   = c[0].selectbox("Asset Model", opts("Asset Model"))
        variant = c[1].selectbox("Asset Variant", sorted(df[df["Asset Model"]==model]["Asset Variant"].astype(str).unique()) or opts("Asset Variant"))
        fuel    = c[2].selectbox("Fuel Type", opts("Asset Fuel Type"))
        cost    = c[3].number_input("Asset Cost (₹)", 40000, 400000, 110000, 5000)
        c = st.columns(4)
        loan    = c[0].number_input("Loan Amount (₹)", 20000, 380000, 95000, 5000)
        tenure  = c[1].selectbox("Tenure (months)", [12,18,24,30,36,42,48], index=5)
        irr     = c[2].number_input("Proposed Rate (IRR %)", 12.0, 34.0, 25.5, 0.25)
        apprisk = c[3].selectbox("App Score Risk", opts("App Score Risk"))
        c = st.columns(4)
        region  = c[0].selectbox("Region", opts("Cust Region"))
        state   = c[1].selectbox("State", opts("Cust State"))
        tier    = c[2].selectbox("Pincode Tier", opts("Pincode Tier"))
        reg     = c[3].selectbox("Registration", opts("Registration Flag"))
        go = st.form_submit_button("Score & recommend", type="primary")

    if go:
        raw = {"Agmt Id":"LIVE_INPUT","Cust Age":35,"Cust Gender":"M","Cust Cibil Score":710,
               "Cust Employment Type":"SAL","Cust Net Salary":28000,"Coborrower Flag":"N","App Score Risk":apprisk,
               "Agmt Date":"2025-06-15","Tenure":int(tenure),"Cust Net IRR":float(irr),"Cust Branch":"BR01",
               "Cust Region":region,"Cust State":state,"Pincode Tier":tier,"RC Availability":"N",
               "Registration Flag":reg,"Asset Disc Flag":1,"Asset Alloy Flag":0,"Asset Variant":variant,
               "Asset Model":model,"Asset Fuel Type":fuel,"Asset Cost At Disbursal":int(cost),
               "Loan Amount":int(loan),"LTV":loan/cost}
        eng = engine.recommend(engine.score_risk(engine.forecast_residual(engine.engineer(raw))))
        row = eng.iloc[0]

        k = st.columns(4)
        k[0].metric("Residual risk", f"{row['Residual_Risk_Score']:.0f}/100", row["Risk_Band"])
        k[1].metric("Recommended LTV", f"{row['Rec_LTV']:.0%}", f"{(row['Rec_LTV']-raw['LTV'])*100:.0f} pts")
        k[2].metric("Recommended rate", f"{row['Rec_Rate']:.2f}%", f"{row['Rec_Rate']-irr:+.2f}")
        k[3].metric("Recommended tenure", f"{int(row['Rec_Tenure'])} mo", f"{int(row['Rec_Tenure'])-int(tenure)}")
        k2 = st.columns(3)
        k2[0].metric("Expected LGD", f"{row['Rec_E_LGD']:.0%}", f"{(row['Rec_E_LGD']-row['Cur_E_LGD'])*100:.0f} pp")
        k2[1].metric("Negative-equity window", f"{int(row['Rec_NegEq_Months'])} mo",
                     f"{int(row['Rec_NegEq_Months'])-int(row['Cur_NegEq_Months'])} mo")
        k2[2].metric("Risk-adjusted net-value change", f"₹{row['NetValue_Lift']:,.0f}")

        left, right = st.columns([3,2])
        with left:
            rv = engine._rv_curve(row); ages = np.arange(1,61)
            nc, nr = int(np.clip(row["Tenure"],6,48)), int(row["Rec_Tenure"])
            OSc = engine._os(raw["LTV"]*cost, irr/1200, nc); OSr = engine._os(row["Rec_LTV"]*cost, row["Rec_Rate"]/1200, nr)
            fig, ax = plt.subplots(figsize=(7,4.2))
            ax.plot(ages, rv, color="green", lw=2.3, label="Forecast residual value")
            ax.plot(ages[:nc], OSc, color=RED, lw=2, label=f"OS – current ({raw['LTV']:.0%}/{irr:.1f}%/{nc}m)")
            ax.plot(ages[:nr], OSr, color=BRAND, lw=2, ls="--", label=f"OS – recommended ({row['Rec_LTV']:.0%}/{row['Rec_Rate']:.1f}%/{nr}m)")
            ax.fill_between(ages[:nc], rv[:nc], OSc, where=OSc>rv[:nc], color=RED, alpha=.15)
            ax.set_xlabel("Asset age (months)"); ax.set_ylabel("₹"); ax.legend(fontsize=7)
            ax.set_title("Amortisation vs residual value — the negative-equity window")
            st.pyplot(fig)
        with right:
            st.markdown("**Residual value forecast**")
            st.table(pd.DataFrame({"Horizon":["12 m","24 m","36 m"],
                "Forecast ₹":[f"₹{row[f'Residual_Value_Forecast_{h}m']:,.0f}" for h in (12,24,36)]}).set_index("Horizon"))
            use_llm = st.toggle("Polish with GenAI (needs ANTHROPIC_API_KEY)", value=False)
            st.markdown("**AI Lending Copilot — rationale**")
            st.info(engine.rationale(eng.iloc[[0]], use_llm=use_llm))

# ================================================================ TAB 3 — SCENARIO SIMULATOR
with tab3:
    st.subheader("Stress the portfolio against market disruption")
    preset = st.radio("Preset", ["Custom","EV Acceleration","Fuel Price Spike","High Inflation","Macro Downturn"],
                      horizontal=True)
    P = {"Custom":dict(ice=0,ev=0,pd=0,cof=0),
         "EV Acceleration":dict(ice=-12,ev=0,pd=5,cof=0),
         "Fuel Price Spike":dict(ice=-8,ev=0,pd=0,cof=0),
         "High Inflation":dict(ice=5,ev=5,pd=10,cof=2),
         "Macro Downturn":dict(ice=-10,ev=-15,pd=30,cof=0)}[preset]
    c = st.columns(4)
    ice = c[0].slider("ICE residual shock (%)", -30, 15, P["ice"])
    ev  = c[1].slider("EV residual shock (%)",  -30, 15, P["ev"])
    pdm = c[2].slider("PD change (%)",           0, 60, P["pd"])
    cof = c[3].slider("Funding-cost add (pp)",   0,  5, P["cof"])

    r = engine.portfolio_scenario(df, RV, rv_ice=1+ice/100, rv_ev=1+ev/100, pd_mult=1+pdm/100, cof_add=cof/100)
    k = st.columns(3)
    k[0].metric("Loss — current policy", f"₹{r['cur_loss_cr']:.1f} Cr")
    k[1].metric("Loss — recommended policy", f"₹{r['rec_loss_cr']:.1f} Cr",
                f"{100*(r['rec_loss_cr']-r['cur_loss_cr'])/r['cur_loss_cr']:.0f}%")
    k[2].metric("Mean LGD (recommended)", f"{r['rec_lgd']*100:.1f}%", f"vs {r['cur_lgd']*100:.1f}% current")
    st.bar_chart(pd.DataFrame({"₹ Cr loss":{"Current policy":r["cur_loss_cr"],"Recommended policy":r["rec_loss_cr"]}}),
                 color=BRAND, horizontal=True)
    st.caption("Because the historical book carries no market-cycle signal (all sales fell in a ~15-month window), "
               "disruption is modelled as an explicit parametric overlay on residual values, PD and funding cost — "
               "and the recommended policy is stress-tested with all levers held fixed.")
