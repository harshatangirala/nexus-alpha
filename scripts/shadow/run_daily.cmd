@echo off
rem Forward shadow test, daily run (registered in Windows Task Scheduler as "NEXUS Alpha shadow test").
rem Forecast first (must finish before 09:00 New York on the next session), then score whatever has matured.
cd /d "%~dp0\..\.."
if not exist artifacts\logs mkdir artifacts\logs
rem Owner-held key file paths (FRED for v2 context) from a git-ignored local file, if present
if exist scripts\shadow\local_env.cmd call scripts\shadow\local_env.cmd
echo ==== %DATE% %TIME% ==== >> artifacts\logs\shadow_daily.log
for %%V in (v1.1 v2) do (
  .venv\Scripts\python.exe -B scripts\shadow\shadow_forecast.py --version %%V >> artifacts\logs\shadow_daily.log 2>&1
  .venv\Scripts\python.exe -B scripts\shadow\shadow_score.py --version %%V >> artifacts\logs\shadow_daily.log 2>&1
)
rem E19 market-direction read (SPY 20D, D-031): logged the same way and scored by the same scorer
.venv\Scripts\python.exe -B scripts\shadow\market_read.py --version market_e19 >> artifacts\logs\shadow_daily.log 2>&1
.venv\Scripts\python.exe -B scripts\shadow\shadow_score.py --version market_e19 >> artifacts\logs\shadow_daily.log 2>&1
rem Statistical process control of the forward test (alerts in the log and shadow/spc_report.json)
.venv\Scripts\python.exe -B scripts\shadow\spc_monitor.py >> artifacts\logs\shadow_daily.log 2>&1
rem Regression alarm: the full test suite every night (summary in the log)
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider -W ignore >> artifacts\logs\nightly_tests.log 2>&1 || echo TESTS FAILED %DATE% >> artifacts\logs\shadow_daily.log
rem Public timestamp: commit and push ONLY the forward-test logs (shadow/), never anything else
git add shadow >> artifacts\logs\shadow_daily.log 2>&1
git diff --cached --quiet -- shadow || (git -c user.name="Krishna Harsha Tangirala" -c user.email="70001399+harshatangirala@users.noreply.github.com" commit -q -m "Forward test: logs of %DATE%" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- shadow >> artifacts\logs\shadow_daily.log 2>&1 && git push -q origin HEAD >> artifacts\logs\shadow_daily.log 2>&1)
