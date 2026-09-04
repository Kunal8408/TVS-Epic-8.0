"""
TVS EPIC 8 — Dynamic Residual Pricing & Lending Strategy Engine (reusable inference module).
Single source of truth for the forecast -> risk -> recommendation -> explanation pipeline.
Imported by the notebooks, the Streamlit dashboard, and the web prototype so they all share ONE validated path.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, joblib

ROOT = Path(__file__).parent
OUT, MOD = ROOT/"outputs", ROOT/"models"

# ---- feature contracts (must match Notebooks 1-3) ----
HORIZON = "Asset Age Months At Seizure"
ASSET_FEATURES = ["Asset Model","Asset Variant","Asset Fuel Type","Is_EV","Asset Disc Flag","Asset Alloy Flag",
                  "Registration Flag","Asset Cost At Disbursal","Cust Region","Cust State","Pincode Tier",
                  "Pincode_Tier_Ord", HORIZON]
COND_MAP = {"G":100,"A":50,"P":0}
RATE_PREMIUM = {"Low":0.0,"Medium":0.25,"High":1.00,"Critical":2.00}
TENURE_POLICY = {"Low":48,"Medium":48,"High":48,"Critical":36}
PD_MAP = {"ULTRA LOW RISK":0.02,"LOW RISK":0.04,"MEDIUM RISK":0.07,"HIGH RISK":0.12,"VERY HIGH RISK":0.18}
COF, BETA_LTV, KAPPA, LTV_FLOOR, LTV_CEIL = 0.09, 1.5, 0.12, 0.65, 0.92
AGES = np.arange(1, 61)

_A = None
def load_assets():
    """Lazy-load models + manifest + risk percentile map + optimal policy params."""
    global _A
    if _A is not None: return _A
    man = json.loads((OUT/"feature_manifest.json").read_text())
    opt = json.loads((OUT/"optimization_summary.json").read_text())
    scored = pd.read_parquet(OUT/"scored_dataset.parquet")
    lgd_ref = np.sort(scored["Predicted_LGD"].values)            # empirical CDF for risk-score percentile
    pmf = np.bincount(np.clip(np.round(scored[HORIZON].values).astype(int),1,60), minlength=61)[1:61].astype(float)
    pmf /= pmf.sum()
    _A = dict(forecaster=joblib.load(MOD/"residual_forecaster.pkl"),
              risk=joblib.load(MOD/"lgd_risk_model.pkl"),
              orig_features=man["origination_features"], lgd_ref=lgd_ref, pmf=pmf,
              ltv_top=opt["assumptions"]["optimal_ltv_top"], slope=opt["assumptions"]["optimal_slope"])
    return _A

# ---------------------------------------------------------------- feature engineering (raw -> model-ready)
def engineer(raw) -> pd.DataFrame:
    df = pd.DataFrame([raw]) if isinstance(raw, dict) else raw.copy()
    if "Traiffic Challan Amount" in df: df = df.rename(columns={"Traiffic Challan Amount":"Traffic Challan Amount"})
    if HORIZON not in df:            df[HORIZON] = 24.0   # placeholder horizon; overwritten per 12/24/36m forecast
    if "Is_EV" not in df:            df["Is_EV"] = (df["Asset Fuel Type"]=="EV").astype(int)
    if "Pincode_Tier_Ord" not in df: df["Pincode_Tier_Ord"] = df["Pincode Tier"].astype(str).str.extract(r"(\d+)").astype(int)
    if "Agmt_Year" not in df and "Agmt Date" in df: df["Agmt_Year"] = pd.to_datetime(df["Agmt Date"]).dt.year
    if "Agmt_Year" not in df:         df["Agmt_Year"] = 2024
    for c in ["Asset Bodycondition","Asset Tyrecondition","Asset Generalcondition","Asset Enginecondition"]:
        if c in df and c+"_score" not in df: df[c+"_score"] = df[c].map(COND_MAP)
    if "Asset_Health_Index" not in df:
        sc = [c+"_score" for c in ["Asset Bodycondition","Asset Tyrecondition","Asset Generalcondition","Asset Enginecondition"] if c+"_score" in df]
        if sc: df["Asset_Health_Index"] = df[sc].mean(axis=1)
    return df

def _cat(X):
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == "object": X[c] = X[c].astype("category")
    return X

# ---------------------------------------------------------------- forecast + risk
def forecast_residual(df):
    A = load_assets(); out = df.copy()
    for h in (12,24,36):
        X = _cat(out[ASSET_FEATURES].copy()); X[HORIZON] = h
        out[f"Residual_Value_Forecast_{h}m"] = (np.clip(A["forecaster"].predict(X),0.05,1.3)*out["Asset Cost At Disbursal"].values).round(0)
    out["Predicted_Residual_Ratio"] = np.clip(A["forecaster"].predict(_cat(out[ASSET_FEATURES])),0.05,1.3)
    return out

def score_risk(df):
    A = load_assets(); out = df.copy()
    out["Predicted_LGD"] = A["risk"].predict(_cat(out[A["orig_features"]]))
    out["Residual_Risk_Score"] = (np.searchsorted(A["lgd_ref"], out["Predicted_LGD"].values)/len(A["lgd_ref"])*100).round(1)
    out["Risk_Band"] = pd.cut(out["Residual_Risk_Score"],[0,25,50,75,100],
                              labels=["Low","Medium","High","Critical"],include_lowest=True).astype(str)
    return out

# ---------------------------------------------------------------- economic engine + policy recommendation
def _rv_curve(row):
    A = load_assets(); base = pd.concat([pd.DataFrame([row[ASSET_FEATURES]])]*len(AGES), ignore_index=True)
    base[HORIZON] = AGES
    return np.clip(A["forecaster"].predict(_cat(base)),0.05,1.3)*float(row["Asset Cost At Disbursal"])
def _os(P,r_m,n):
    t=AGES[:n]; f=(1+r_m); return P*(f**n - f**t)/(f**n - 1)
def _evaluate_one(row, ltv, rate, n, rv, pmf, pd_base, cur_ltv):
    P=ltv*float(row["Asset Cost At Disbursal"]); r_m=rate/1200.0; OS=_os(P,r_m,n); RVn=rv[:n]
    w=pmf[:n]/pmf[:n].sum(); short=np.clip(OS-RVn,0,None)
    e_lgd=float((np.clip(short/np.clip(OS,1,None),0,1)*w).sum()); ead=float((OS*w).sum())
    pne=float(short.max()); nem=int((short>0).sum())
    pd_eff=float(np.clip(pd_base*(1+BETA_LTV*(ltv-cur_ltv)),0.003,0.9))
    emi=P*r_m*(1+r_m)**n/((1+r_m)**n-1) if r_m>0 else P/n
    net=(emi*n-P)*(1-pd_eff) - P*COF*(n/12.0)*0.5 - pd_eff*e_lgd*ead - KAPPA*pne
    return dict(E_LGD=e_lgd, EAD=ead, NegEq_Months=nem, Net=net)
def recommend(df):
    A = load_assets(); out = df.copy()
    if "Residual_Risk_Score" not in out: out = score_risk(out)
    recs = []
    for _, row in out.iterrows():
        cur_ltv = float(np.clip(row["LTV"],0.5,1.0)); cur_rate=float(row["Cust Net IRR"]); cur_ten=int(np.clip(row["Tenure"],6,48))
        band = row["Risk_Band"]; pd_base = PD_MAP.get(row.get("App Score Risk"),0.07)
        rv = _rv_curve(row)
        rec_ltv  = float(np.clip(A["ltv_top"]-A["slope"]*row["Residual_Risk_Score"]/100.0, LTV_FLOOR, LTV_CEIL))
        rec_rate = float(np.clip(cur_rate+RATE_PREMIUM.get(band,0),12,34))
        rec_ten  = int(min(cur_ten, TENURE_POLICY.get(band,48)))
        cur = _evaluate_one(row,cur_ltv,cur_rate,cur_ten,rv,A["pmf"],pd_base,cur_ltv)
        rec = _evaluate_one(row,rec_ltv,rec_rate,rec_ten,rv,A["pmf"],pd_base,cur_ltv)
        recs.append(dict(Rec_LTV=round(rec_ltv,3),Rec_Rate=round(rec_rate,2),Rec_Tenure=rec_ten,
                         Cur_E_LGD=round(cur["E_LGD"],3),Rec_E_LGD=round(rec["E_LGD"],3),
                         Cur_NegEq_Months=cur["NegEq_Months"],Rec_NegEq_Months=rec["NegEq_Months"],
                         NetValue_Lift=round(rec["Net"]-cur["Net"],0)))
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(recs)], axis=1)

# ---------------------------------------------------------------- explainability (SHAP) + NL rationale
def explain_row(df_row_frame, which="risk", top=4):
    """Return top SHAP drivers (feature, value, signed contribution) for one loan."""
    import shap
    A = load_assets()
    model = A["risk"] if which=="risk" else A["forecaster"]
    feats = A["orig_features"] if which=="risk" else ASSET_FEATURES
    X = _cat(df_row_frame[feats])
    sv = shap.TreeExplainer(model).shap_values(X)
    contrib = pd.Series(np.asarray(sv)[0], index=feats)
    idx = contrib.abs().sort_values(ascending=False).head(top).index
    return [(f, df_row_frame.iloc[0][f], float(contrib[f])) for f in idx]

_HUMAN = {
 "Asset Model":"asset model {v}", "Asset Variant":"variant {v}", "Pincode_Tier_Ord":"location tier {v} (higher = more rural, thinner resale)",
 "Asset Cost At Disbursal":"asset price ₹{v:,.0f}", "Cust Cibil Score":"CIBIL {v}", "Cust Net IRR":"current rate {v:.1f}%",
 "LTV":"LTV {v:.0%}", "Tenure":"tenure {v} months", "App Score Risk":"application risk '{v}'", "Is_EV":"EV flag {v}",
 "Cust Net Salary":"net salary ₹{v:,.0f}", "Cust Age":"age {v}", "Cust Region":"region {v}", "Cust State":"state {v}",
 "Asset Fuel Type":"fuel {v}", "Registration Flag":"registration {v}"}
def _phrase(f, v):
    if isinstance(v, float): v = round(v, 1)
    try: return _HUMAN.get(f, f+" {v}").format(v=v)
    except Exception: return f"{f} {v}"

def rationale(rowf, use_llm=False):
    """Deterministic, SHAP-grounded plain-English recommendation for underwriters.
    If use_llm=True and an API key + SDK are available, an LLM rewrites it more fluently; otherwise this text stands."""
    r = rowf.iloc[0]
    risk_drv = explain_row(rowf, "risk", 3)
    val_drv  = [d for d in explain_row(rowf, "forecaster", 5) if d[0] != HORIZON][:3]  # horizon is the axis, not a driver
    up = lambda d: ", ".join(_phrase(f,v) for f,v,c in d)
    f12,f24,f36 = (int(r.get(f"Residual_Value_Forecast_{h}m",0)) for h in (12,24,36))
    txt = (
      f"RECOMMENDATION for {r['Agmt Id']} ({r['Asset Model']}) — risk band {r['Risk_Band']} "
      f"(score {r['Residual_Risk_Score']:.0f}/100).\n"
      f"• Terms: set LTV to {r['Rec_LTV']:.0%} (from {float(r['LTV']):.0%}), price at {r['Rec_Rate']:.2f}% "
      f"(from {float(r['Cust Net IRR']):.2f}%), cap tenure at {int(r['Rec_Tenure'])} months (from {int(r['Tenure'])}).\n"
      f"• Residual outlook: forecast to fetch ₹{f12:,} / ₹{f24:,} / ₹{f36:,} at 12 / 24 / 36 months. "
      f"Main value drivers: {up(val_drv)}.\n"
      f"• Risk rationale: expected loss-given-default falls from {float(r['Cur_E_LGD']):.0%} to {float(r['Rec_E_LGD']):.0%}, "
      f"and the negative-equity window narrows from {int(r['Cur_NegEq_Months'])} to {int(r['Rec_NegEq_Months'])} months. "
      f"Key risk drivers: {up(risk_drv)}.\n"
      f"• Economic impact: risk-adjusted net-value change ₹{float(r['NetValue_Lift']):,.0f}."
    )
    if use_llm:
        polished = _llm_polish(txt)
        if polished: return polished
    return txt

def _llm_polish(text):
    """Optional GenAI layer. Activates ONLY if ANTHROPIC_API_KEY is set and the SDK is installed. Never raises."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"): return None
    try:
        import anthropic
        c = anthropic.Anthropic()
        m = c.messages.create(model="claude-sonnet-5", max_tokens=400,
            system="You are a credit underwriting copilot. Rewrite the recommendation as a crisp, professional "
                   "2-3 sentence note for a loan officer. Keep every number exactly as given. Do not invent facts.",
            messages=[{"role":"user","content":text}])
        return m.content[0].text
    except Exception:
        return None

def score_and_recommend(raw, explain=True, use_llm=False):
    df = recommend(score_risk(forecast_residual(engineer(raw))))
    res = {"recommendation": df.iloc[[0]].to_dict("records")[0]}
    if explain: res["rationale"] = rationale(df.iloc[[0]], use_llm=use_llm)
    return res

# ================================================================ VECTORISED PORTFOLIO ENGINE (dashboard / simulator)
def build_rv_matrix(df):
    """Residual-value curve for every loan over ages 1..60 (asset-intrinsic; independent of loan terms)."""
    A = load_assets(); cost = df["Asset Cost At Disbursal"].values.astype(float)
    base = _cat(df[ASSET_FEATURES].copy()); RV = np.zeros((len(df), 60))
    for j, age in enumerate(range(1, 61)):
        X = base.copy(); X[HORIZON] = age; RV[:, j] = np.clip(A["forecaster"].predict(X), 0.05, 1.3) * cost
    return RV

def _os_mat(P, r_m, n):
    t = np.arange(1, n+1); f = (1+r_m)[:, None]
    return P[:, None] * (f**n - f**t[None, :]) / (f**n - 1)

def portfolio_eval(df, RV, ltv, rate, ten, pd_base, cof=COF):
    """Vectorised economic engine over the whole book. Returns per-loan Net, E_LGD, GDloss (₹), NegEqMonths."""
    A = load_assets(); pmf = A["pmf"]; N = len(df)
    cost = df["Asset Cost At Disbursal"].values.astype(float); cur_ltv = df["LTV"].clip(0.5, 1.0).values
    net = np.zeros(N); el = np.zeros(N); gl = np.zeros(N); nem = np.zeros(N, int)
    P = ltv*cost; r_m = rate/1200.0
    pd_eff = np.clip(pd_base*(1+BETA_LTV*(ltv-cur_ltv)), 0.003, 0.9)
    for n in np.unique(ten):
        m = ten == n; nn = int(n)
        OS = _os_mat(P, r_m, nn); RVn = RV[:, :nn]; w = pmf[:nn]/pmf[:nn].sum()
        short = np.clip(OS-RVn, 0, None); lgd_t = np.clip(short/np.clip(OS, 1, None), 0, 1)
        e = (lgd_t*w[None, :]).sum(1); ead = (OS*w[None, :]).sum(1); pne = short.max(1); nm = (short > 0).sum(1)
        emi = np.where(r_m > 0, P*r_m*(1+r_m)**nn/((1+r_m)**nn-1), P/nn)
        nt = (emi*nn-P)*(1-pd_eff) - P*cof*(nn/12.0)*0.5 - pd_eff*e*ead - KAPPA*pne
        for arr, val in zip((net, el, gl, nem), (nt, e, e*ead, nm)): arr[m] = val[m]
    return dict(Net=net, E_LGD=el, GDloss=gl, NegEqMonths=nem)

def recommended_terms(df):
    """Vectorised recommended LTV / rate / tenure under the optimised risk-based policy."""
    A = load_assets()
    score = df["Residual_Risk_Score"].values; band = df["Risk_Band"].astype(str).values
    cur_ltv = df["LTV"].clip(0.5, 1.0).values; cur_rate = df["Cust Net IRR"].values
    cur_ten = df["Tenure"].clip(6, 48).astype(int).values
    rec_ltv = np.clip(A["ltv_top"]-A["slope"]*score/100.0, LTV_FLOOR, LTV_CEIL)
    rec_rate = np.clip(cur_rate + pd.Series(band).map(RATE_PREMIUM).fillna(0).astype(float).values, 12, 34)
    rec_ten = np.minimum(cur_ten, pd.Series(band).map(TENURE_POLICY).fillna(48).astype(int).values)
    return rec_ltv, rec_rate, rec_ten

def portfolio_scenario(df, RV, rv_ice=1.0, rv_ev=1.0, pd_mult=1.0, cof_add=0.0):
    """Apply a parametric shock and compare current vs recommended policy across the book."""
    is_ev = (df["Is_EV"].values == 1)
    RVs = RV.copy(); RVs[~is_ev] *= rv_ice; RVs[is_ev] *= rv_ev
    pd_base = np.clip(df["App Score Risk"].map(PD_MAP).fillna(0.07).values*pd_mult, 0, 0.9)
    cur_ltv = df["LTV"].clip(0.5, 1.0).values; cur_rate = df["Cust Net IRR"].values
    cur_ten = df["Tenure"].clip(6, 48).astype(int).values
    rec_ltv, rec_rate, rec_ten = recommended_terms(df)
    cur = portfolio_eval(df, RVs, cur_ltv, cur_rate, cur_ten, pd_base, COF+cof_add)
    rec = portfolio_eval(df, RVs, rec_ltv, rec_rate, rec_ten, pd_base, COF+cof_add)
    return dict(cur_loss_cr=cur["GDloss"].sum()/1e7, rec_loss_cr=rec["GDloss"].sum()/1e7,
                cur_lgd=float(cur["E_LGD"].mean()), rec_lgd=float(rec["E_LGD"].mean()),
                cur_negeq=float(cur["NegEqMonths"].mean()), rec_negeq=float(rec["NegEqMonths"].mean()))
