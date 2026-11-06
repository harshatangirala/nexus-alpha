"""The nightly runner (Task Scheduler -> run_daily.cmd) must be byte-clean and reference only existing scripts.

Audit 4 (2026-11-07): a patch wrote "\\v" (a vertical tab) into a script path in run_daily.cmd; cmd.exe would have failed
that step silently every night."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CMD = REPO / "scripts" / "shadow" / "run_daily.cmd"


def test_runner_bytes_are_clean_crlf():
    b = CMD.read_bytes()
    assert not [c for c in set(b) if c < 32 and c not in (10, 13)], "control characters in run_daily.cmd"
    assert b.count(b"\n") == b.count(b"\r\n"), "run_daily.cmd must use CRLF throughout"


def test_runner_scripts_exist():
    text = CMD.read_text(encoding="ascii")
    refs = re.findall(r"scripts\\[\w\\]+\.py", text)
    assert refs, "no script references found"
    for r in refs:
        assert (REPO / Path(*r.split("\\"))).exists(), f"run_daily.cmd references a missing script: {r}"


def test_every_logged_version_is_frozen_and_monitored():
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "shadow"))
    import spc_monitor
    text = CMD.read_text(encoding="ascii")
    scored = set(re.findall(r"shadow_score\.py --version ([\w.]+)", text)) | {"v1.1", "v2"}
    for v in scored:
        assert (REPO / "registry" / "frozen_models" / f"{v}.json").exists(), v
        assert v in spc_monitor.UNIVERSE, f"{v} is scored nightly but not in the SPC monitor"
