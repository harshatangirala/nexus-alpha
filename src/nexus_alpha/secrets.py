"""Load API keys from owner-held .env files into the process environment, never into the repository or logs.

The owner keeps keys outside the repository (e.g. Downloads/FRED_API.env, Downloads/Gemini_API.env) or sets them as user
environment variables. `load()` reads every file listed in NEXUS_ENV_FILES (separated by ';') and sets only names that
are not already set. Values are never printed; `status()` reports only which names are present.
"""
from __future__ import annotations

import os
from pathlib import Path

NAMES = ("FRED_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "SEC_USER_AGENT")


def load(files: str | None = None) -> list[str]:
    files = files if files is not None else os.environ.get("NEXUS_ENV_FILES", "")
    loaded = []
    for f in [x for x in files.split(";") if x.strip()]:
        p = Path(f.strip())
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in NAMES and v and not os.environ.get(k):
                os.environ[k] = v
                loaded.append(k)
    return loaded


def status() -> dict[str, bool]:
    return {k: bool(os.environ.get(k)) for k in NAMES}
