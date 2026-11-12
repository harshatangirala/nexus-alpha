# External integrations (D-046 to D-050, 2026-11-10)

The owner asked for OpenBB, TradingAgents and Qlib to make the engine more solid, and for six visual projects to make
the dashboard stronger. This file records what each one adds, how it is integrated, and what it cannot do.

## The rule behind every choice

The project's evidence comes from three things: pre-registered tests with placebos, the point-in-time policy, and a
forward test that no development step could see. An integration is useful only if it keeps those intact.
1. **Nothing changes a frozen model or the forward test's inputs.** New code is additive and runs in new modules.
2. **A new predictive input enters only through a pre-registered experiment with a placebo**, under the existing keep
   rule, on validation only.
3. **An LLM's output never enters a backtest.** A modern LLM has read about the outcomes of 2003-2022, so any
   historical score of it measures memory, not skill (D-020). It can only be tested forward.
4. **Data without point-in-time vintages is live context only**, never a training input.
5. **Heavy tools get their own environment** (`artifacts/envs/`, git-ignored; `src/nexus_alpha/envs.py`), called
   through subprocesses that exchange files. The environment the frozen models run in imports none of them.

## Status by repository

| Repository | Licence | What it adds | Integration | Status |
|---|---|---|---|---|
| microsoft/qlib | MIT | Alpha158, the most used public factor library, as a benchmark for the 80 hand-built features | Re-implemented in `src/nexus_alpha/indicators/alpha158.py` (157 of 158 features; VWAP0 needs a VWAP the bars lack). Checked against pyqlib 0.9.7's own engine on 23 stocks and 122,797 stock-days: identical definitions. Tested as **E25**. `v3_alpha` frozen and in the forward test at 5D-1Y (D-049) | done |
| TauricResearch/TradingAgents | Apache-2.0 | A multi-agent LLM committee: analysts, a bull/bear debate, a trader, a risk team, a portfolio manager | v0.7.1 in `artifacts/envs/tradingagents`; `src/nexus_alpha/llm/committee.py`; dashboard tab M; forward scoring **E27** | done |
| OpenBB (`openbq-org/OpenBB`, the Open Data Platform; the old `OpenBB-finance` address redirects there) | Apache-2.0 | One interface to many data providers; here CBOE's delayed option chains | 5.0.0 in `artifacts/envs/openbb`; `scripts/openbb/options_chains.py`; `src/nexus_alpha/options.py`; dashboard tab N; nightly snapshot and forward benchmark **E26** | done |
| vasturiano/globe.gl | MIT | WebGL globe | Mission Control globe: company addresses and Global Macro Database country data (pinned 2.46.2) | done |
| arwes/arwes | MIT | Framed sci-fi UI language | Recreated in CSS (`app/ui.py`, `.streamlit/config.toml`) | done |
| magicuidesign/magicui | MIT | Animated components | NumberTicker, Marquee and AnimatedBeam behaviour, re-written in plain HTML/CSS/JS | done |
| manelalab/chrono-bert-v1-20101231 (Hugging Face) | MIT | A chronologically consistent language model (He, Lv, Manela and Wu, 2025): pretrained only on text before 2011, so it cannot leak the evaluation years | `src/nexus_alpha/nlp/transformer_text.py` (revision pinned); encodes 68,853 SEC filings for the `sec_tx` block; dashboard tab O; notebook section 8A. Tested as **E28** | done |
| SEC filings price-impact study (the owner's project) | own | 190,947 point-in-time filing events, their text and the price-impact labels | Read only: events and text since v1 (`data/sec.py`); from E28 also its `T_trade` labels, learned by the SEC price-impact expert (`nlp/sec_impact.py`, without `meta_year` and the accounting families, review findings N1/N2) | done |
| koala73/worldmonitor | AGPL-3.0 | A situational-awareness layout around a globe | Layout reference only. AGPL code cannot be copied into this repository | done |
| hal9ai/awesome-dataviz, sjfricke/awesome-webgl | lists | Library choice | Plotly and deck.gl (pydeck) were already installed; globe.gl is the WebGL globe | done |

## Qlib: E25 and v3_alpha

* **Pre-registration.** `docs/V3_RESEARCH.md` (E25), written before the features were built.
* **Result.** Kept on ranking only; no gain in probabilities; partial 2020-22 replication (D-047).
* **Parity.** Qlib itself is never imported by the package. The parity check exports the sealed bars in Qlib's binary
  format and runs Qlib's expression engine in `artifacts/envs/qlib`. Qlib computes some inputs in float32, which
  explains the only differences (at most 3.1e-5, in the correlation features); with float32 inputs those agree on
  every row too.
* **v3_alpha.** Refitted, frozen and given a live-parity check against Yahoo bars, whose criteria were written before
  it ran:
  * price features pass (median correlation 0.9989);
  * rank_score agreement passes at 5D-1Y and fails at 1D/2D, because Yahoo's daily volume differs from the vendor's.
  It is therefore logged at 5D-1Y only, from session 2026-11-09 (D-049).

## TradingAgents: the LLM committee (E27)

* **What it does.** Dashboard tab M runs TradingAgents v0.7.1 for one stock as of the last closed session, on the
  owner's Gemini key. It shows the analysts' reports, the debate and the five-tier rating, and puts the frozen v1.1
  lean next to it.
* **Isolation.** The key travels only through the child process environment, never through a file. TradingAgents'
  result folders, cache and memory log live in `artifacts/llm_committee/work/`, not in the user's home folder.
* **Logging.** Every run is recorded twice:
  * the full reports privately (`artifacts/llm_committee/runs/`; LLM text built from news and filings);
  * the rating with its metadata and the reports' SHA-256 publicly (`shadow/committee/`, hash-chained). The nightly
    job pushes that record to GitHub, which fixes its time.
* **Scoring.** Forward only, by `scripts/shadow/committee_score.py` under E27's pre-registered rules: one vote per
  stock per 5 sessions, no claim before 100 runs. A run after the decision time is flagged late and never scored.
* **Never on history.** TradingAgents' own backtester is not used as evidence (rule 3).
* **Quota.** The free Gemini tier allows 20 requests per model per day, shared with the Trade Desk's Gemini read. A
  market-only run takes about 12 calls by our estimate from the graph's steps (each analyst adds about 4).
  * The adapter refuses a run that would not fit what is left today.
  * It adds every call it made (counted by a callback) to the shared usage counter.
  * This is why the committee runs on demand and never nightly.
* **Live status, 2026-11-10.** The integration is complete and tested (quota guard, model fallback, public record, forward scorer). The first live run could not complete: Google answered 503 "high demand" on all four flash models available to the key, and a direct one-word request was refused too. Nothing was logged. The first completed run is still to come, from tab M.

## OpenBB: the options market's view (E26)

* **Dashboard tab N.** A stock's option chain from CBOE's delayed quotes, with:
  * the 30-day ATM implied volatility, the front expiry and put/call open interest;
  * the options-implied 1D/5D/20D moves and 10%-90% ranges, next to v1.1's own logged return range;
  * the ATM term structure.
* **Nightly snapshot** (`scripts/shadow/iv_snapshot.py`). Each night, after the forecasts, every forward-test stock's
  chain is captured and reduced to the E26 measures (`src/nexus_alpha/options.py`, definitions fixed by the
  pre-registration).
  * The data stays private (`artifacts/iv_snapshots/`) under CBOE's terms.
  * Each snapshot's counts, times and SHA-256 are hash-chained publicly in `shadow/iv_snapshot/`.
* **The test** (`scripts/shadow/iv_benchmark_score.py`, statistics in `src/nexus_alpha/evaluation/iv_benchmark.py`):
  after 120 sessions, does NEXUS's P(move) add to an implied-volatility-only baseline? The statistics were checked on
  synthetic data: they find information that is there and reject noise.
* **Engineering findings (2026-11-10).**
  * **Command layer too slow.** `obb.cboe.options.chains` took about 40 s per stock under threads. The runner calls
    the same OpenBB provider code (`openbb_cboe`'s `CboeOptionsChainsFetcher`: its extract and transform steps)
    directly, at about 2 s per stock.
  * **Outdated certificate store.** The interpreter's default store failed CBOE's (valid, Let's Encrypt)
    certificate. The runner points `SSL_CERT_FILE` at certifi's current bundle; verification stays on.
  * **Throttling.** CBOE's CDN throttles concurrent bursts (82 of 500 stocks at 8 requests at a time). The runner
    fetches one stock at a time with retries and backoff, about 15 minutes for 500 stocks.
  * **An OpenBB transform bug.** OpenBB's transform raises on some payloads (TSLA: a missing last-trade time). A
    minimal fallback parser reads the same raw payload. It agrees exactly with OpenBB's transform on every contract
    of AAPL, KO, JPM and ZUMZ (6,360 contracts, zero differences in bid, ask, IV, days to expiry and spot).
  * **Coverage.** Stocks missing from CBOE's options directory (for example NVR, AHT) have no IV and are reported as
    missing.

## What these integrations cannot fix

* **Survivorship.** The universe is today's survivors from one vendor. None of these projects supplies free
  historical delisted constituents.
* **The reused test years.** 2020-22 has been read many times. Only the forward test and the sealed 2023+ period are
  clean.
* **Sample size.** E26 and E27 are forward tests: months of nightly data before they can say anything. No library
  shortens that.
