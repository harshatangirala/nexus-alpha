@echo off
rem Forward shadow test, daily run (registered in Windows Task Scheduler as "NEXUS Alpha shadow test").
rem Forecast first (must finish before 09:00 New York on the next session), then score whatever has matured.
cd /d "%~dp0\..\.."
if not exist artifacts\logs mkdir artifacts\logs
echo ==== %DATE% %TIME% ==== >> artifacts\logs\shadow_daily.log
for %%V in (v1.1) do (
  .venv\Scripts\python.exe -B scripts\shadow\shadow_forecast.py --version %%V >> artifacts\logs\shadow_daily.log 2>&1
  .venv\Scripts\python.exe -B scripts\shadow\shadow_score.py --version %%V >> artifacts\logs\shadow_daily.log 2>&1
)
