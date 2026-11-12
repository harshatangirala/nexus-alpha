"""The nightly keep-awake helper must watch the batch (cmd.exe), not the venv launcher (A-63, audit 6).

The venv's python.exe is a launcher that runs the real interpreter as its child. A helper that watched its direct
parent would watch that launcher, which lives as long as the helper does, and would hold the PC awake for its whole
safety limit. Found by running the helper under a real batch before it went into run_daily.cmd.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.name != "nt", reason="Windows batch behaviour")
def test_helper_watches_the_batch_not_the_venv_launcher(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("import sys\n"
                     f"sys.path.insert(0, r'{REPO / 'scripts' / 'shadow'}')\n"
                     "import keep_awake as K\n"
                     "print(K.batch_pid())\n", encoding="utf-8")
    p = subprocess.Popen(["cmd", "/c", sys.executable, "-B", str(probe)], stdout=subprocess.PIPE, text=True)
    out, _ = p.communicate(timeout=120)
    assert int(out.strip().splitlines()[-1]) == p.pid          # the cmd.exe that ran the line


def test_wrap_mode_releases_and_returns_the_exit_code():
    r = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "shadow" / "keep_awake.py"), "--",
                        sys.executable, "-c", "raise SystemExit(3)"], timeout=120)
    assert r.returncode == 3
