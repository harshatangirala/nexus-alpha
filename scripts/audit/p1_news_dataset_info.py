"""Phase 1: metadata audit of the Hugging Face news dataset (no article text is downloaded here).

Records gating, licence, card metadata, configs and the data-file list with sizes into
data_manifests/news_dataset_info.json. Uses the locally cached Hugging Face login if one exists; the token is
never printed or written.

Run: .venv/Scripts/python -I scripts/audit/p1_news_dataset_info.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = Path(__file__).resolve().parents[2]
DATASET = "Brianferrell787/financial-news-multisource"


def main() -> int:
    api = HfApi()
    out = {"dataset": DATASET, "queried_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        info = api.dataset_info(DATASET, files_metadata=True)
    except Exception as e:  # access blocker is a result, not a crash
        out["error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(out, indent=1))
        (REPO / "data_manifests" / "news_dataset_info.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        return 1
    out.update({
        "sha": info.sha,
        "last_modified": str(info.last_modified),
        "private": info.private,
        "gated": info.gated,
        "disabled": getattr(info, "disabled", None),
        "downloads": getattr(info, "downloads", None),
        "tags": info.tags,
        "card_data": info.card_data.to_dict() if info.card_data else None,
    })
    files = []
    for s in info.siblings or []:
        files.append({"path": s.rfilename, "bytes": getattr(s, "size", None),
                      "lfs_sha256": (s.lfs.sha256 if getattr(s, "lfs", None) else None)})
    out["n_files"] = len(files)
    out["total_bytes"] = sum(f["bytes"] or 0 for f in files)
    out["files"] = files
    try:
        readme = hf_hub_download(DATASET, "README.md", repo_type="dataset")
        out["readme_text"] = Path(readme).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        out["readme_error"] = f"{type(e).__name__}: {e}"
    (REPO / "data_manifests").mkdir(exist_ok=True)
    (REPO / "data_manifests" / "news_dataset_info.json").write_text(json.dumps(out, indent=1, default=str),
                                                                   encoding="utf-8")
    summary = {k: v for k, v in out.items() if k not in ("files", "readme_text", "card_data")}
    print(json.dumps(summary, indent=1, default=str))
    by_ext = {}
    for f in files:
        ext = f["path"].rsplit(".", 1)[-1] if "." in f["path"] else ""
        by_ext.setdefault(ext, [0, 0])
        by_ext[ext][0] += 1
        by_ext[ext][1] += f["bytes"] or 0
    print("files by extension:", {k: (n, round(b / 1e9, 3)) for k, (n, b) in by_ext.items()})
    for f in files[:60]:
        print(f"  {f['path']:<90} {(f['bytes'] or 0) / 1e6:>10.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
