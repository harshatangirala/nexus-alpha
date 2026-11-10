# External integrations (D-046, 2026-11-10)

The owner asked for OpenBB, TradingAgents and Qlib to make the engine more solid, and for six visual projects to make
the dashboard stronger. This file records what each one adds, how it is integrated, and what is left for later.

## The rule behind every choice

The project's evidence comes from three things: pre-registered tests with placebos, the point-in-time policy, and a
forward test that no development step could see. An integration is useful only if it keeps those intact.
1. **Nothing changes a frozen model or the forward test's inputs.** New code is additive and runs in new modules.
2. **A new predictive input enters only through a pre-registered experiment with a placebo**, under the existing keep
   rule, on validation only.
3. **An LLM's output never enters a backtest.** A modern LLM has read about the outcomes of 2003-2022, so any
   historical score of it measures memory, not skill (D-020). It can only be tested forward.
4. **Data without point-in-time vintages is live context only**, never a training input.
5. **Heavy tools get their own environment** (`artifacts/envs/`, git-ignored), so the environment the frozen models
   run in does not change.

## Status by repository

| Repository | Licence | What it adds | Integration | Status |
|---|---|---|---|---|
| microsoft/qlib | MIT | Alpha158, the most used public factor library, as a benchmark for the 80 hand-built features | Re-implemented in `src/nexus_alpha/indicators/alpha158.py` (157 of 158 features; VWAP0 needs a VWAP the bars lack). Checked against pyqlib 0.9.7's own engine on 23 stocks and 122,797 stock-days: the definitions are identical (`reports/verification/alpha158_qlib_parity.json`). Tested as **E25** under the E22/E23 keep rule | done; result below |
| TauricResearch/TradingAgents | Apache-2.0 | A multi-agent LLM committee: analysts, a bull/bear debate, a trader, a risk team | A live, on-demand second opinion in the Trade Desk, logged for forward scoring (plan below) | next step |
| OpenBB (`openbq-org/OpenBB`, the Open Data Platform; the old `OpenBB-finance` address redirects there) | Apache-2.0 | One interface to many data providers | Live context, and an implied-volatility benchmark for the forward test (plan below) | next step |
| vasturiano/globe.gl | MIT | WebGL globe | Mission Control globe: company addresses and Global Macro Database country data (pinned 2.46.2) | done |
| arwes/arwes | MIT | Framed sci-fi UI language | Recreated in CSS (`app/ui.py`, `.streamlit/config.toml`) | done |
| magicuidesign/magicui | MIT | Animated components | NumberTicker, Marquee and AnimatedBeam behaviour, re-written in plain HTML/CSS/JS | done |
| koala73/worldmonitor | AGPL-3.0 | A situational-awareness layout around a globe | Layout reference only. AGPL code cannot be copied into this repository | done |
| hal9ai/awesome-dataviz, sjfricke/awesome-webgl | lists | Library choice | Plotly and deck.gl (pydeck) were already installed; globe.gl is the WebGL globe | done |

## Qlib: E25 Alpha158 benchmark

Pre-registered in `docs/V3_RESEARCH.md` (section E25) before the features were built; result there and in
`docs/DECISION_LOG.md`. Qlib itself is never imported by the package: the parity check exports the sealed bars in
Qlib's binary format and runs Qlib's expression engine in `artifacts/envs/qlib`. Qlib computes some inputs in
float32, which explains the only differences (at most 3.1e-5, in the correlation features); with float32 inputs
those agree on every row too.

## TradingAgents: plan (not built yet)

**What it would be.** A button in the Trade Desk, "Ask the LLM committee". It runs TradingAgents v0.7.1 (pinned) in
its own environment on the owner's Gemini key and shows the analysts' reports, the debate and the five-tier rating.
The rating is never blended into the model's probabilities and never touches the risk engine. Every run is appended
to a hash-chained log (as the Gemini reads are), so its ratings can be scored forward.

**The quota fact that shapes it.** One analysis with three analysts, one debate round and one risk round makes
roughly 15-30 model calls. That is our estimate from the graph's steps, not a published figure: each analyst's tool
rounds, two researchers, the research manager, the trader, three risk analysts and the portfolio manager. The
Gemini free tier allows **20 requests per model per day** (recorded in
`src/nexus_alpha/llm/gemini.py`), and the Trade Desk's own Gemini read uses the same quota. So on the free tier, one
committee run can use up a model's day. The choices:
* give the committee a different Gemini model from the Trade Desk's (for example a "lite" model for the analysts), so
  their quotas do not collide; or
* use a paid Gemini tier; or
* run a local model through Ollama (free, slower, weaker).

**Forward test (a draft, to be pre-registered before the first logged run).**
* A fixed panel: for example the 20 largest forward-test stocks by market value, fixed before the first run.
* A weekly run.
* Ratings mapped Buy/Overweight = up, Hold = sideways, Underweight/Sell = down. TradingAgents' REVIEW signal
  (output with no readable rating) is logged and counted as a failure, never scored as Hold.
* Scored at 5D and 20D with the frozen models' label rule, and compared with v1.1 on the same stock-days.
* About 120 scored sessions before any claim.

TradingAgents' own backtester is not used as evidence (rule 3).

**Steps.**
1. `uv venv artifacts/envs/tradingagents --python 3.13`.
2. Clone at tag v0.7.1 and run `uv pip install .`.
3. Pass the owner's key as `GOOGLE_API_KEY` at run time only.
4. Write an adapter (`src/nexus_alpha/llm/committee.py`) that calls it through a subprocess with a timeout.
5. Run one smoke test, then the Trade Desk panel and its tests.

About half a day.

## OpenBB: plan (not built yet)

**Install.** `openbb` 5.0.0 (Python 3.10+) in `artifacts/envs/openbb`, called by a small script that writes JSON or
parquet.

**Uses, in order of value.**
1. **Implied-volatility benchmark (the most important test it enables).** The models' one solid skill is "will it
   move". The options market prices exactly that. Each night, log the 30-day at-the-money implied volatility of each
   forward-test stock next to the forecasts, before 09:00 New York. After about 120 sessions, test whether P(move)
   from NEXUS adds anything to an implied-volatility-only baseline on sideways-versus-move log loss. If it does not,
   the volatility skill is the options market's, already priced. Pre-register this before the first snapshot.
2. **A second data source for the audit.** Compare the SEC-derived fundamentals of the largest stocks with OpenBB's
   providers, automatically, as a repeatable check.
3. **Live context in the Trade Desk.** Insider filings (Form 4), short interest (FINRA) and the economic calendar.
   These are context only: rule 4.

About a day for item 1 (the nightly snapshot needs the same hardening as the forecasters), half a day each for 2 and 3.

## What these integrations cannot fix

* **Survivorship.** The universe is today's survivors from one vendor. None of these projects supplies free
  historical delisted constituents.
* **The reused test years.** 2020-22 has been read many times. Only the forward test and the sealed 2023+ period are
  clean.
* **Sample size.** The forward test needs months, not days. No library shortens that.
