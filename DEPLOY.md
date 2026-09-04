# Deploying to Streamlit Community Cloud (free, ~5 minutes)

Netlify/Vercel/GitHub Pages **cannot** run Streamlit (it needs a live Python server). Streamlit Community Cloud is purpose-built and free.

## Step 1 — Put this folder in a GitHub repo

**Option A — GitHub website (no command line):**
1. Create a free account at github.com and click **New repository** (name it e.g. `tvs-residual-engine`). Public is simplest; private also works.
2. On the repo page, click **Add file -> Upload files**, then drag in EVERY file and folder from this `streamlit_app_deploy` folder (including `models/`, `outputs/`, `.streamlit/`).
3. Click **Commit changes**.

**Option B — Git command line (run inside this folder):**
```bash
git init
git add .
git commit -m "TVS residual pricing Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/tvs-residual-engine.git
git push -u origin main
```

## Step 2 — Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io** and sign in with your GitHub account.
2. Click **Create app -> Deploy a public app from GitHub**.
3. Fill in: **Repository** = your repo, **Branch** = `main`, **Main file path** = `app.py`.
4. (Recommended) Open **Advanced settings** and set **Python version = 3.11**.
5. Click **Deploy**. First build takes ~2-4 minutes.

You'll get a public URL like `https://your-app-name.streamlit.app` — share it in your submission.

## Notes & troubleshooting
- `models/` and `outputs/` **must** be committed to the repo (they're small, a few MB total) — the app loads them at startup. Do not add them to `.gitignore`.
- If the build fails on a specific package, loosen that line in `requirements.txt` (e.g. change `shap==0.49.1` to `shap>=0.44`), commit, and Streamlit auto-redeploys.
- The app needs no secrets or API keys. (An optional GenAI wording-polish activates only if you add an `ANTHROPIC_API_KEY` in Streamlit **Settings -> Secrets**, but it is not required.)
- To update the live app later, just push new commits to the repo — it redeploys automatically.
