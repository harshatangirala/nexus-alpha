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
