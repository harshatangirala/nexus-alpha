@echo off
rem Forward shadow test, daily run (registered in Windows Task Scheduler as "NEXUS Alpha shadow test").
rem Forecast first (must finish before 09:00 New York on the next session), then score whatever has matured.
rem Audit 3: Yahoo preflight (waits up to 90 minutes); with extra daily triggers a later run only proceeds if an earlier one did not complete.
cd /d "%~dp0\..\.."
if not exist artifacts\logs mkdir artifacts\logs
rem Owner-held key file paths (FRED for v2 context) from a git-ignored local file, if present
if exist scripts\shadow\local_env.cmd call scripts\shadow\local_env.cmd
echo ==== %DATE% %TIME% ==== >> artifacts\logs\shadow_daily.log
rem A-63 (D-052): keep Windows out of idle sleep while this batch runs; the helper exits when the batch ends
start "" /b .venv\Scripts\python.exe -B scripts\shadow\keep_awake.py --watch-parent
rem Preflight: Yahoo reachable? (exit 10 = already completed today, 2 = unreachable after about 90 minutes)
.venv\Scripts\python.exe -B scripts\shadow\preflight.py >> artifacts\logs\shadow_daily.log 2>&1
if errorlevel 10 exit /b 0
if errorlevel 2 exit /b 2
set FOK=1
for %%V in (v1.1 v2) do (
  .venv\Scripts\python.exe -B scripts\shadow\shadow_forecast.py --version %%V >> artifacts\logs\shadow_daily.log 2>&1 || set FOK=0
  .venv\Scripts\python.exe -B scripts\shadow\shadow_score.py --version %%V >> artifacts\logs\shadow_daily.log 2>&1
)
rem E19 market-direction read (SPY 20D, D-031): logged the same way and scored by the same scorer
.venv\Scripts\python.exe -B scripts\shadow\market_read.py --version market_e19 >> artifacts\logs\shadow_daily.log 2>&1 || set FOK=0
.venv\Scripts\python.exe -B scripts\shadow\shadow_score.py --version market_e19 >> artifacts\logs\shadow_daily.log 2>&1
rem Live fundamentals cross-section for the Trade Desk (live inference only; SEC at <= 5 requests/s; non-fatal)
.venv\Scripts\python.exe -B scripts\shadow\live_fundamentals.py >> artifacts\logs\shadow_daily.log 2>&1
rem v3_fund (E23 kept, D-042): F0 rows from tonight's v2 run + the live fundamentals; scored like the others
.venv\Scripts\python.exe -B scripts\shadow\v3_fund_forecast.py >> artifacts\logs\shadow_daily.log 2>&1
.venv\Scripts\python.exe -B scripts\shadow\shadow_score.py --version v3_fund >> artifacts\logs\shadow_daily.log 2>&1
rem v3_alpha (E25 kept, D-049): F0 rows from tonight's v2 run + Alpha158 from tonight's cached bars; 5D-1Y only
.venv\Scripts\python.exe -B scripts\shadow\v3_alpha_forecast.py >> artifacts\logs\shadow_daily.log 2>&1
.venv\Scripts\python.exe -B scripts\shadow\shadow_score.py --version v3_alpha >> artifacts\logs\shadow_daily.log 2>&1
rem Forward promotion criteria of v3_fund and v3_alpha vs ranker_v3, Holm-corrected (D-051); progress until 120 sessions
.venv\Scripts\python.exe -B scripts\shadow\forward_criteria.py >> artifacts\logs\shadow_daily.log 2>&1
rem E26 (D-050): option-implied volatility snapshot via OpenBB/CBOE (data private; SHA-256 chained); non-fatal
.venv\Scripts\python.exe -B scripts\shadow\iv_snapshot.py >> artifacts\logs\shadow_daily.log 2>&1
.venv\Scripts\python.exe -B scripts\shadow\iv_benchmark_score.py >> artifacts\logs\shadow_daily.log 2>&1
rem E27 (D-050): forward scoring of on-demand LLM committee runs; non-fatal
.venv\Scripts\python.exe -B scripts\shadow\committee_score.py >> artifacts\logs\shadow_daily.log 2>&1
rem Statistical process control of the forward test (alerts in the log and shadow/spc_report.json)
.venv\Scripts\python.exe -B scripts\shadow\spc_monitor.py >> artifacts\logs\shadow_daily.log 2>&1
rem Regression alarm: the full test suite every night (summary in the log)
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider -W ignore >> artifacts\logs\nightly_tests.log 2>&1 || echo TESTS FAILED %DATE% >> artifacts\logs\shadow_daily.log
rem Public timestamp: commit and push ONLY the forward-test logs (shadow/), never anything else
git add shadow >> artifacts\logs\shadow_daily.log 2>&1
git diff --cached --quiet -- shadow || (git -c user.name="Krishna Harsha Tangirala" -c user.email="70001399+harshatangirala@users.noreply.github.com" commit -q -m "Forward test: logs of %DATE%" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- shadow >> artifacts\logs\shadow_daily.log 2>&1 && git push -q origin HEAD >> artifacts\logs\shadow_daily.log 2>&1)
if "%FOK%"=="1" .venv\Scripts\python.exe -B scripts\shadow\preflight.py --mark >> artifacts\logs\shadow_daily.log 2>&1
rem Audit 11 (D-059): live headline archive for a later forward test of the text branch (E29b). Runs last, after the
rem push and the done marker, so it can never delay a forecast; git-ignored output (artifacts\live_news); non-fatal
.venv\Scripts\python.exe -B scripts\shadow\news_snapshot.py >> artifacts\logs\shadow_daily.log 2>&1
