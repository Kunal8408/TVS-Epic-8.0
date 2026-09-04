# Dynamic Residual Pricing & Lending Strategy Engine

TVS Credit — EPIC 8 Analytics Challenge 2026. An interactive Streamlit app that forecasts a financed two-wheeler's residual value, scores residual risk, and recommends optimal lending terms (LTV / rate / tenure).

## Three tools in one app
- **Dynamic Pricing Dashboard** — portfolio impact, risk bands, segment risk, LTV shift.
- **AI Lending Copilot** — price a single loan and get a plain-English rationale.
- **Scenario Simulator** — stress the book against EV / fuel / inflation / downturn shocks.

It is also the browser-based **web prototype**. All figures are produced by `engine.py`, the same validated pipeline used in the analysis notebooks.

## Run locally
```bash
pip install -r requirements.txt   # Python 3.10+
streamlit run app.py
```
Opens at http://localhost:8501 . No internet or API key required.

## Deploy
See `DEPLOY.md` for one-click hosting on Streamlit Community Cloud.
