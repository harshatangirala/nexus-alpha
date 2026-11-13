# Forward shadow test

Started 2026-11-06 on the owner's instruction, with the frozen v1.1 model. v2 joins when it is frozen.

**What it is.** Every evening after the US close, each frozen model forecasts all 500 universe stocks from live Yahoo bars.
The forecasts are logged before the next session opens and are scored only after their horizons end. This is the only
test of direction skill that no part of model development could have seen.

**Integrity.**
* `registry/frozen_models/<version>.json` pins the SHA-256 of every bundle file; a run refuses to start if anything changed.
* `shadow/<version>/forecasts/<asof>.csv` is written once (first write wins) and appended to `shadow/<version>/chain.csv`,
  a hash chain over (run time, file SHA-256, previous row hash). Editing any logged forecast breaks the chain, and the
  scorer refuses to score a broken chain. Committing `shadow/` to git adds a public timestamp.
* A row counts only if it was written before its decision time (09:00 New York on the next weekday). Later rows are kept,
  flagged `late`, and never scored.

**Scoring.** `scripts/shadow/shadow_score.py` applies the training label rule (entry at the t+1 open, exit at the t+h
close, threshold max(0.5 sigma sqrt h, 10 bps)) to Yahoo bars and reports, per horizon: log loss against the training
class prior, up-vs-down AUROC, argmax hit rate, and a date-clustered bootstrap interval once 20 dates have matured.

**Running it.**
```
scripts\shadow\run_daily.cmd            # forecast, then score; logs to artifacts\logs\shadow_daily.log
```
Automatic daily runs need a Windows scheduled task, which the owner registers (the assistant is not permitted to create
persistent system tasks). In PowerShell:
```
$a = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$PWD\scripts\shadow\run_daily.cmd`""
$t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday,Wednesday,Thursday,Friday,Saturday -At 07:00
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 3)
Register-ScheduledTask -TaskName "NEXUS Alpha shadow test" -Action $a -Trigger $t -Settings $s
```
(run from the repository folder; 07:00 IST is after every US close in both daylight-saving regimes and well before the
09:00 New York decision). Remove with `Unregister-ScheduledTask -TaskName "NEXUS Alpha shadow test"`.

**How long until it says anything.** Roughly 20 matured dates for a first interval at 1D, about two months at 20D. Direction
skill of the size seen on validation (AUROC 0.52-0.54) needs on the order of a year of daily forecasts to separate from
0.5 with confidence.

## Changes on 2026-11-06 (after the first session)

* **What is logged.** Three frozen models are logged every evening, all starting with session 2026-11-05:
  * v1.1;
  * v2 (`registry/frozen_models/v2.json`);
  * the E19 SPY 20D market read (`market_e19`, `scripts/shadow/market_read.py`).
* **Same scorer.** All three are scored by `shadow_score.py` with each record's horizons and training class prior.
* **Robustness (D-028).** The forecaster has:
  * a 180 s timeout per ticker;
  * retry passes that wait longer only when at least 10 tickers come back empty;
  * a stop 10 minutes before the decision time.
* **Bars.** The bars fetched in one New York evening are cached and shared by v1.1 and v2, so both forecast from identical inputs.
* **Keys.** The task reads the paths of the owner's key files from the git-ignored `scripts/shadow/local_env.cmd`. v2's FRED context needs it (audit A-27).
* **Public timestamps.** After scoring, the task commits **only `shadow/`** and pushes it to the public repository. That fixes a public time for each forecast before its outcome is known.
* **No news.** In live mode the news modality is always missing for both stock models: there is no live news feed, and texts reach the models only through the Trade Desk.

## Changes on 2026-11-07 (audit 3 and 4)

* **A fourth frozen model, `v3_fund`** (D-042; E23 kept). It is LightGBM on the v1.1 numeric features plus
  industry-neutral fundamentals, refitted on 2009-2018.
  * **Inputs:** v2's assembled feature row for each stock (the v2 forecasts are identical with and without this extra
    output, A/B tested) and the same session's live fundamentals cross-section (`live_fundamentals.py`: SEC facts at
    <= 5 requests/s with the owner's declared User-Agent).
  * **Log:** `shadow/v3_fund/` with `rank_score`, scored by `shadow_score.py`. No trading engine uses it.
  * **Promotion:** the forward criterion in `docs/V3_RESEARCH.md` (20D rank IC above 0 and above `ranker_v3` after 120
    sessions).
* **Preflight and circuit breaker (D-041, D-055).** Before forecasting, the task checks two things, waiting up to 90
  minutes for both: Yahoo answers, and at least 6 of 8 liquid stocks already have the session the run must log. The
  second check was added after the 13 Nov run (A-80), when SPY was current but stock downloads were a day behind.
  A forecast run whose first 20 tickers all fail stops at once instead of running into the 3-hour task limit.
  The run's deadline is the expected session's decision time, so one stale stock cannot stop it (A-81).
  A done marker, keyed by session, lets the extra daily triggers re-run only an incomplete session.
* **Runner tests.** `tests/test_shadow_runner.py` checks three things:
  * `run_daily.cmd` is byte-clean CRLF;
  * every script it references exists;
  * every version it scores is frozen and monitored by SPC.

  Audit 4 found a vertical-tab byte that a patch had written into a script path.
* **Kronos (`kronos_zs`, D-043/D-044): not logged.** E24's necessary condition was not met, so it never joins the run.
  The tooling below stays in the repository, inactive:
  * **Forecaster:** `scripts/shadow/kronos_forecast.py`. It uses only tonight's cached bars, takes about 24 minutes
    for 500 stocks on CPU, and is hash-checked through a flat bundle.
  * **When:** it joins `run_daily.cmd` only if E24's pre-registered necessary condition is met on 2017-18.
  * **Promotion:** the forward criterion in `docs/V3_RESEARCH.md`.

## Incident on 2026-11-07 (A-50, D-045)

* **What happened.** The 07:00 IST run reused a bar cache filled before Friday's close. v1.1 and v2 forecast the
  already-logged session 2026-11-05 and wrote nothing; the run still marked itself complete; v3_fund logged a 1-stock
  file for 2026-11-05 with every row late.
* **Fixes.** A run now targets `expected_asof()` (the last session whose 16:00 New York close plus 5 minutes has
  passed). A cache without that session is refetched. The done marker needs v1.1, v2 and market_e19 to have logged it.
  v3_fund refuses partial or stale input. The scorer skips an all-late version.
* **Makeup.** The full sequence re-ran at 13:05 IST on 2026-11-07, before the 2026-11-09 09:00 New York decision time:
  v1.1, v2 and v3_fund each logged 499 stocks for 2026-11-06 with no late rows; every chain is intact.
* **Permanent record.** The junk v3_fund file stays in the append-only log. It is never scored (late), and the SPC
  monitor lists it as acknowledged, pinned by its SHA-256, instead of raising a nightly alarm.

## Incident on 2026-11-10 (D-048)

* **What happened.** The 07:00 IST run for session 2026-11-09 did not start on time: the PC was not awake, and the
  task has no wake timer (`WakeToRun` is off). Task Scheduler started it when the PC woke, at 10:19 IST. It ended
  with exit code 0xC000013A (the console was closed, or Ctrl+C) before writing its first log line: the task runs as an
  interactive task in a visible console window. A manual start at 13:58 was stopped with Ctrl+C after its first line.
* **Makeup.** The full nightly sequence ran from 14:21 IST at the owner's instruction and finished at 16:04 IST,
  before the decision time (09:00 New York on 2026-11-10 = 19:30 IST):
  * v1.1 logged 496 stocks x 6 horizons;
  * v2 logged 496 x 23;
  * market_e19 logged its SPY read;
  * v3_fund logged 496 x 9.

  No row is late, every chain is intact, and only `shadow/` was committed and pushed, as every night.
* **Also found and fixed (A-53).** `scripts/shadow/local_env.cmd` had a corrupted last line. The keys still loaded.
* **Owner actions recommended** (the assistant does not change scheduled tasks):
  1. Turn on "Wake the computer to run this task" (Task Scheduler, Conditions tab). Wake timers must be allowed in
     the power plan, and the PC must be asleep, not shut down.
  2. Run the task without a console window: General tab, "Run whether user is logged on or not" (asks for the
     Windows password once). Then the window cannot be closed by accident.

## Changes on 2026-11-10 (D-049): v3_alpha joins

* **What it is.** A fifth frozen model, `v3_alpha`: LightGBM on the v1.1 numeric features plus Qlib's Alpha158 (E25
  kept on rank IC), refitted on 2003-2018.
* **Horizons.** It is logged at 5D-1Y only. Its live-parity check (Yahoo against vendor bars on 2022) passed there and
  failed at 1D/2D (`reports/verification/alpha158_live_parity.json`).
* **Inputs.** v2's assembled F0 row (as for v3_fund) and the Alpha158 features of the as-of session, computed from
  the same cached Yahoo bars v1.1 and v2 used that night (`scripts/shadow/v3_alpha_forecast.py`).
* **Start.** First logged session 2026-11-09, written on 2026-11-10 at 17:18 IST, before that session's decision time.
* **Monitoring.** Scored by `shadow_score.py` and monitored by the SPC monitor like the others. Not needed for the done
  marker, as for v3_fund.
* **Forward criterion.** As for v3_fund: 20D rank IC above 0 and above `ranker_v3` after 120 sessions.

## Changes on 2026-11-10 (D-050): option snapshots and committee records

* **Option snapshot (E26).** After the forecasts, `iv_snapshot.py` captures every forward-test stock's option chain
  through OpenBB (CBOE delayed quotes).
  * It fetches one stock at a time, because CBOE throttles bursts. 500 stocks take about 15 minutes.
  * It reduces each chain to the E26 measures.
  * The data stays private. The public record `shadow/iv_snapshot/<asof>.json` (counts, times, both files' SHA-256)
    is hash-chained.
  * `iv_benchmark_score.py` scores E26 once sessions mature. It decides nothing before 120 sessions at 20D.
* **Committee records (E27).** Each on-demand LLM committee run from the dashboard adds a public record to
  `shadow/committee/`, hash-chained like the forecasts. `committee_score.py` scores them nightly.
* **Effect on the other models.** Both steps are non-fatal, run after every model has logged, and do not count
  towards the done marker.

## Incident on 2026-11-11/12 (D-052, A-63)

* **What happened.**
  * **11 Nov.** The laptop uses Modern Standby. The 07:00 IST run for session 2026-11-10 was missed while it slept
    (no wake timer). Task Scheduler started it when the laptop woke at 13:58. The run wrote its header line 10
    seconds after the wake and then nothing more: every step failed before printing.
  * **12 Nov.** At 09:09 a Windows Update restart (System event 1074) ended the next attempt with 0xC0000142. Its
    processes could not start while the session was shutting down.
  * After the restart the task could not run until the owner logged on: it runs only while the user is logged on.
* **Session 2026-11-10 is missed.** Its decision time (09:00 New York on 11 Nov = 19:30 IST) passed with nothing
  logged. It is not back-filled: a late file would be flagged and never scored. Mission Control now shows each
  model's missed sessions (`missed_sessions` in `app/mission_control.py`).
* **Make-up of session 2026-11-11.** The unchanged nightly runner ran from 14:32 IST on 12 Nov, wrapped by
  `scripts/shadow/keep_awake.py` so the laptop could not idle into standby. It finished at 16:25 IST, before the
  decision time (19:30 IST):
  * v1.1 logged 497 stocks x 6 horizons;
  * v2 logged 497 x 23;
  * market_e19 logged its SPY read;
  * v3_fund logged 497 x 9;
  * v3_alpha logged 497 x 7.

  No row is late and every chain is intact. The option snapshot (E26) covered 362 of 500 stocks. The nightly commit
  `1117150` reached GitHub at about 16:25 IST, before the decision time. Only `shadow/` was committed and pushed, as
  every night.
* **Changes (infrastructure only; no forecast is affected).**
  * `run_daily.cmd` starts the keep-awake helper, which holds off idle sleep while the batch runs and exits when it
    ends.
  * The done marker is keyed by the as-of session, not the local date (A-64). An extra trigger after midnight can no
    longer make the 07:00 run skip a session. That makes extra triggers safe.
* **Owner actions** (the assistant does not change scheduled tasks). Run `scripts\shadow\owner_fix_task.ps1` in an
  administrator PowerShell. It:
  * turns on the wake timer and allows wake timers in the power plan;
  * runs a missed start as soon as possible, on battery too;
  * adds log-on and unlock triggers;
  * can switch the task to "run whether user is logged on or not".

  Keep the laptop plugged in and asleep, not shut down, overnight.
