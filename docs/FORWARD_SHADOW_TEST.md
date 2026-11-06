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
