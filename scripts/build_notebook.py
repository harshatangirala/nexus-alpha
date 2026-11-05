"""Build notebooks/NEXUS_Alpha_Colab_Orchestrator.ipynb (thin front end over the package; no logic lives in cells).
Run: python scripts/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# NEXUS Alpha: Colab orchestrator

Runs the NEXUS Alpha pipeline on Google Colab with Google Drive as the persistent store. Research and paper trading only.

* Secrets come from **Colab Secrets** (`HF_TOKEN`); they are never printed or written to files.
* The sealed period (from 2023-01-03) is never read. The final evaluation needs the owner's key (`docs/POINT_IN_TIME_POLICY.md`).
* Every stage skips work whose output already exists on Drive, so a disconnect loses at most the current stage; fusion
  training also checkpoints every epoch.
* Default profile `smoke_test` only runs the automated tests. Heavier profiles print a cost estimate and need an explicit opt-in.
"""),
    md("## 1. Environment and hardware"),
    code("""
import subprocess, sys, os
print(sys.version)
try:
    import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
                       torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except ImportError:
    print("torch not installed yet")
!nvidia-smi --query-gpu=name,memory.total --format=csv || true
!df -h /content | tail -1
"""),
    md("## 2. Mount Drive and point NEXUS at it\nEdit `DRIVE_ROOT` if your folders differ. The repository and the source data must already be on Drive (the TriFrame bars and caches are private and are not downloaded from GitHub)."),
    code("""
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/antigravity_workspace'          # edit if needed
REPO = f'{DRIVE_ROOT}/nexus-alpha'
os.environ['NEXUS_ARTIFACTS'] = f'{DRIVE_ROOT}/nexus_artifacts'      # persistent: never rely on /content alone
os.environ['NEXUS_OHLCV_ROOT'] = f'{DRIVE_ROOT}/us_stocks_ohlcv'
os.environ['NEXUS_SEC_FULL500_RUN'] = f'{DRIVE_ROOT}/sec_inputs_staging/full'
os.environ['NEXUS_DOWNLOADS'] = DRIVE_ROOT
os.environ['NEXUS_TRIFRAME_REPO'] = f'{DRIVE_ROOT}/triframe-cv'
os.makedirs(os.environ['NEXUS_ARTIFACTS'], exist_ok=True)
%cd {REPO}
"""),
    md("## 3. Install (pinned versions in `requirements-colab.txt`)"),
    code("""
!pip -q install -r requirements-colab.txt
!pip -q install -e .
"""),
    md("## 4. Secrets (never printed)"),
    code("""
from google.colab import userdata
try:
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    print('HF_TOKEN loaded from Colab Secrets')
except Exception:
    print('No HF_TOKEN secret: private TriFrame embeddings and the gated news corpus will not be readable')
"""),
    md("## 5. Preflight: automated tests and hardware report"),
    code("""
!python scripts/run_pipeline.py --profile smoke_test
"""),
    md("## 6. Choose the profile\n`research` reproduces the local run (s500 frame, cached TriFrame embeddings). `full` needs a GPU and re-embeds all 979 stocks (not yet implemented as a stage: see docs/IMPLEMENTATION_STATUS.md)."),
    code("""
RUN_PROFILE = 'research'      # 'debug' | 'small' | 'research' | 'full'
ACCEPT_COST = False           # set True only after reading the estimate printed below
!python scripts/run_pipeline.py --profile {RUN_PROFILE} --stages audit
"""),
    md("## 7. Data preparation (embeddings, samples, news, SEC, rule features)"),
    code("""
flag = '--i-accept-the-cost' if ACCEPT_COST else ''
!python scripts/run_pipeline.py --profile {RUN_PROFILE} --stages embeddings,samples,news_index,news_text,text_features,rule_features {flag}
"""),
    md("## 8. Branch baselines (structured, chart branch)"),
    code("""
!python scripts/run_pipeline.py --profile {RUN_PROFILE} --stages structured,cv_branch {flag}
"""),
    md("## 9. Multimodal fusion, calibration, registered hypotheses, inference bundle"),
    code("""
!python scripts/run_pipeline.py --profile {RUN_PROFILE} --stages fusion {flag}
"""),
    md("## 10. Walk-forward evaluation and economic backtest"),
    code("""
!python scripts/run_pipeline.py --profile {RUN_PROFILE} --stages walk_forward {flag}
import glob
e03 = sorted(glob.glob(os.environ['NEXUS_ARTIFACTS'] + '/bundles/*'))[-1].split('/')[-1]
!python -I scripts/experiments/e04_backtest.py {e03}
"""),
    md("## 11. Knowledge base and retrieval evaluation"),
    code("""
!python scripts/run_pipeline.py --profile {RUN_PROFILE} --stages rag {flag}
"""),
    md("## 12. Minimal end-to-end inference (historical mode, sealed data)"),
    code("""
from nexus_alpha.inference.pipeline import NexusPredictor
from nexus_alpha.rag.explain import load_kb
p = NexusPredictor(kb=load_kb())
res = p.predict('AAPL', asof='2021-06-15', mode='historical')
print(res.model_dump_json(indent=1)[:3000])
"""),
    md("## 13. GPU-only extensions (registered, not run in the local profile)\n* Transformer text encoders (FinBERT, BGE, SEC-BERT, ChronoBERT) replacing TF-IDF features: adapters listed in `configs/nlp_models.yaml`.\n* Full-universe re-rendering and re-embedding at native resolution (TriFrame review P1).\nRun these only after reviewing the cost estimate and with results labelled as new experiments."),
]


def main():
    nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "notebooks" / "NEXUS_Alpha_Colab_Orchestrator.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
