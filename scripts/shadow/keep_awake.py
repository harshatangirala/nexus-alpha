r"""Keep Windows out of idle sleep while the nightly run works (incident of 2026-11-11, D-052).

The laptop uses Modern Standby. On 2026-11-10 and 2026-11-11 the scheduled run started seconds after the machine woke
and then produced nothing; on 2026-11-12 a Windows Update restart killed the next attempt (0xC0000142). While this
helper holds its request, Windows does not enter idle sleep (SetThreadExecutionState with ES_SYSTEM_REQUIRED, the call
`powercfg /requests` lists). The request ends when the helper exits. A closed lid or the power button still forces
standby: the owner's task settings (wake timer, run whether logged on or not) remain the real fix.

Two modes:
    python scripts/shadow/keep_awake.py -- cmd /c scripts\shadow\run_daily.cmd   # wrap a command (manual make-up run)
    python scripts/shadow/keep_awake.py --watch-parent                          # from run_daily.cmd via `start /b`

It never touches a forecast, a log of the forward test or the scheduled task.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "artifacts" / "logs" / "keep_awake.log"

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _note(msg: str) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except OSError:
        pass


def hold() -> bool:
    """Ask Windows to stay awake until release() or process exit. False off Windows or if refused."""
    if os.name != "nt":
        return False
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))


def release() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def batch_pid() -> int:
    """PID of the batch (cmd.exe) that started this helper. The venv's `python.exe` is a launcher that runs the real
    interpreter as its child, so the direct parent is usually that launcher, which lives as long as this process does.
    Watching it would hold the PC awake until the safety limit (found in the audit-6 test)."""
    try:
        import psutil
        for p in psutil.Process().parents():
            if p.name().lower() == "cmd.exe":
                return p.pid
        parent = psutil.Process().parent()
        if parent is not None and parent.name().lower().startswith("python"):
            grand = parent.parent()
            if grand is not None:
                return grand.pid
    except Exception:  # noqa: BLE001  (psutil missing or access denied: fall back to the direct parent)
        pass
    return os.getppid()


def alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k = ctypes.windll.kernel32
    h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return bool(ok) and code.value == STILL_ACTIVE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-parent", action="store_true", help="hold while the parent process (the batch) runs")
    ap.add_argument("--max-hours", type=float, default=4.0, help="safety limit")
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run after --")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not a.watch_parent and not cmd:
        ap.error("give --watch-parent or a command after --")
    ok = hold()
    deadline = time.monotonic() + a.max_hours * 3600
    try:
        if cmd:
            _note(f"hold={ok} wrap {' '.join(cmd)}")
            rc = subprocess.call(cmd, cwd=REPO)
            _note(f"released; command exit {rc}")
            return rc
        parent = batch_pid()
        _note(f"hold={ok} watching batch pid {parent}")
        while alive(parent) and time.monotonic() < deadline:
            time.sleep(a.poll)
        _note("released; parent finished" if not alive(parent) else "released; safety limit reached")
        return 0
    finally:
        release()


if __name__ == "__main__":
    sys.exit(main())
