# User guide

## 1. Set up (Windows, local)

```powershell
cd C:\Users\harsh\nexus-alpha
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -r requirements-colab.txt
uv pip install --python .venv\Scripts\python.exe -e .
.venv\Scripts\python.exe -m pytest            # about 75 tests; source-data tests skip without the TriFrame store
```

The source projects are read in place (read-only). Default locations are in `src/nexus_alpha/paths.py`; override with `NEXUS_*` environment variables (see `.env.example`). A Hugging Face login with access to the private TriFrame repositories and the gated news corpus is needed for the data stages.

## 2. Reproduce the research run

```powershell
.venv\Scripts\python.exe scripts\run_pipeline.py --profile research --stages all --i-accept-the-cost
.venv\Scripts\python.exe -I scripts\experiments\e04_backtest.py <E03 run id>
```
Stages skip existing outputs. The full local run took several hours on a laptop CPU, mostly downloads (embeddings 15 min, news text ~1 h over an unstable connection), rule evaluation (13 min) and fusion training (~2 h).

## 3. Run the app

**Easiest:** double-click `RUN_DASHBOARD.cmd` in the repository folder. It starts the dashboard and opens
`http://localhost:8501` in the browser; close its window to stop it. The same from a terminal:

```powershell
.venv\Scripts\streamlit run app\streamlit_app.py
```
**Compare 1-5** (third tab): pick up to five stocks and a logged forward-test session. Everything the nightly run
logged appears at once:
* price paths and drawdowns;
* v1.1/v2 probabilities, return ranges and the lean across horizons;
* a risk-versus-rank map, a factor radar, correlations, a scoreboard of every frozen model and option-implied
  volatility.

"Run the full model and trade plans" then scores each stock live and draws its plan with the reasons.

**Desk 1-5** (fourth tab): your own inputs for one to five stocks at once.
1. Pick the stocks, live or historical data, the trade type, and your time zone (IST by default).
2. In each stock's inputs tab, add up to three chart screenshots (weekly, daily, hourly). Upload a file, or click a
   "paste" box and press Ctrl+V after copying a screenshot (Win+Shift+S on Windows).
3. Add 1-7 news articles per stock, each with its date, time and time zone ("Fetch Yahoo headlines" fills them in UTC).
   Untick "time known" when you only know the date. Optionally add an SEC filing per stock and, in the last tab, macro
   texts shared by every stock.
4. Press "Analyse". You get a board (probabilities, lean, return range, decision, entry, stop-loss, targets, time
   exit), side-by-side charts, the trade plans with their reasons, a chronology chart of every text against the as-of
   close, the 5-day news window and the decision time, and each stock in detail (plan chart, texts in time order,
   screenshot checks, optional Gemini read and ChronoBERT analogues of a pasted filing).

How your inputs are used: the model renders its own charts from real bars and never reads screenshot pixels; the
legend values you type (or Gemini reads) check a screenshot against real bars. Company news enters the model only if it
was public before the decision time (09:00 New York, the session after the as-of close) and inside the 5-day window the
model was trained with. SEC filing and macro texts are context: timed and scored, not model inputs.

**Vision + NLP** (tab O): the E28 result and how much each input feeds the network. Below it:
* the SEC price-impact expert's view of every filing of a stock;
* a filing explorer that finds the most similar earlier filings of other firms (point in time);
* a box where any pasted filing or news text is encoded with ChronoBERT and compared with every encoded filing.

The first paste loads the model, which can take a minute. Everything in the tab is descriptive: it changes no
probability and no plan.

**Levels and reasons.** The Trade Desk, the Trade planner and Compare draw each plan on the price chart with a reason
card. The card covers:
* the entry, stop and target rules with their rulebook ids;
* the weekly regime, daily setup and hourly trigger;
* the indicators, the model's probabilities, news, filings and book citations.

When the risk engine says NO_TRADE, the setup's own levels are drawn dashed and marked CONDITIONAL, with the gates
that failed. The decision itself is unchanged.

Sidebar: pick a ticker from the 500-stock universe, a mode and, in historical mode, an as-of session (2004 to 2022; later dates are sealed). Optionally paste 1-7 articles with their publication time (UTC) and upload screenshots (shown only; never parsed for prices). Set the paper risk settings and press **Run analysis**.

* **Historical mode** uses the sealed TriFrame bars. Dates before 2016 are in-sample for the model, 2017-2019 were used for selection and calibration, 2020-2022 is the development test.
* **Live mode** downloads 3 years of daily and 60 days of hourly bars from Yahoo Finance and scores the last completed session. SEC filings are not fetched live; the result says so.

The first tab, **Mission Control**, needs no analysis run. It reads the repository each time it loads:
* **Header:** the counts of frozen models, last night's stocks, forward sessions, experiments, logged decisions and
  tests, and a tape of what the evidence supports and does not.
* **Forward shadow test:** each frozen model's sessions, rows, late rows and hash-chain check, with progress towards
  the first interval (20 matured dates) and the 120-session criterion.
* **System status:** the last nightly run, the SPC monitor, the sealed period and the reproduction status.
* **Evidence board:** the key results with their 95% intervals, against the no-skill line. Every number quotes its
  source document (`tests/test_dashboard.py` checks that the text is there).
* **Data flow:** how a raw bar becomes a logged forecast.
* **Globe:** where the forward-test companies have their business address, over a Global Macro Database variable.
  This is context only.
* **Qlib benchmark (E25):** the Alpha158 test's keep decision, its gains with intervals, and Qlib-style IC tables.

The globe and the fonts load from public CDNs, so they need an internet connection; everything else works offline.
The theme is in `.streamlit/config.toml`, and the visual sources are listed in `THIRD_PARTY_NOTICES.md`.

## 4. Reading the output

* **Probabilities** are calibrated per horizon on 2019 data. "Direction skill validated" is False when the model's up-versus-down ranking on 2017-2018 was not reliably better than chance. "Replicated 2020-22" is False when the same test on 2020-2022 was not better than chance. The decision engine trades a horizon's direction only when both are True (D-057). Today no horizon of v1.1 or v2 qualifies, so every plan is NO TRADE; the dashed CONDITIONAL levels still show what a setup would look like.
* **NO TRADE** with reasons is the normal, intended output. **NO SIGNAL** means the data were insufficient for numeric levels.
* **Diagnostics** (leave-one-modality-out, chart occlusion) describe this model's sensitivity, not causes.
* **RAG citations** point to a rulebook line or a book page; they explain concepts and never change probabilities.

## 5. Colab

Open `notebooks/NEXUS_Alpha_Colab_Orchestrator.ipynb`, set `DRIVE_ROOT`, add `HF_TOKEN` in Colab Secrets, and run the cells in order. Results are written to Drive, never only to the Colab disk.

## 6. The sealed final evaluation (owner only)

Freeze the configuration file, compute its SHA-256, set `NEXUS_HOLDOUT_CONFIG` and `NEXUS_HOLDOUT_UNLOCK`, then run the final-evaluation script (to be written once the owner decides what to freeze). The unlock is single-use and logged with a hash chain in `governance/holdout_access_log.csv`. Results on 2023+ are semi-confirmatory (that period was opened before by TriFrame and is the SEC study's spent locked test).

### Tabs M and N (live tools, 2026-11-10)

* **M LLM committee.**
  * Pick a stock and the analysts, and press **Run the committee**. It runs TradingAgents on your Gemini key, which
    takes a few minutes.
  * The free tier allows 20 calls a day per model. A market-only run takes about 12, so the tab shows what is left
    and refuses a run that would not fit.
  * Each run is logged publicly and scored forward (E27). A run after 09:00 New York on the session after the as-of
    session is flagged late and never scored.
  * The rating never changes a forecast or the risk engine.
* **N Options.**
  * **Fetch the option chain** for a stock: implied volatility, implied moves and the term structure from CBOE's
    delayed quotes, next to the model's own range.
  * The status line shows the nightly snapshots for the E26 benchmark.

Both tools run in their own environments under `artifacts/envs/`. Install them with `uv`:
* TradingAgents: `uv venv artifacts/envs/tradingagents --python 3.12`, then
  `uv pip install --python artifacts/envs/tradingagents/Scripts/python.exe "tradingagents @ git+https://github.com/TauricResearch/TradingAgents@v0.7.1"`.
* OpenBB: `uv venv artifacts/envs/openbb --python 3.12`, then
  `uv pip install --python artifacts/envs/openbb/Scripts/python.exe openbb==5.0.0`.
