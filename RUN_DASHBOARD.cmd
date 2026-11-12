@echo off
rem NEXUS Alpha dashboard: double-click this file. It starts Streamlit and opens http://localhost:8501 in the browser.
rem Close this window to stop the dashboard. Research and paper trading only; nothing is ever sent to a broker.
cd /d "%~dp0"
if not exist .venv\Scripts\streamlit.exe (
  echo The project environment .venv was not found. See docs\USER_GUIDE.md, section "Install".
  pause
  exit /b 1
)
rem Owner-held key file PATHS (Gemini, FRED, SEC user agent) from the git-ignored local file, if present
if exist scripts\shadow\local_env.cmd call scripts\shadow\local_env.cmd
echo.
echo   NEXUS Alpha is starting at  http://localhost:8501
echo   The browser opens by itself. If it does not, open the address above.
echo   Close this window to stop the dashboard.
echo.
.venv\Scripts\streamlit.exe run app\streamlit_app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
pause
