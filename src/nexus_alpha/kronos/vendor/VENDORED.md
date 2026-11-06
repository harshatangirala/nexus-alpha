# Vendored: Kronos model code

* Source: https://github.com/shiyu-coder/Kronos, commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a` (2026-04-13), files `model/kronos.py` and
  `model/module.py`. Licence: MIT (`LICENSE` in this folder, copyright (c) 2025 ShiYu).
* Paper: Shi et al., *Kronos: A Foundation Model for the Language of Financial Markets*, arXiv:2508.02739 (AAAI 2026).
* SHA-256 of the original files:
  * `model/kronos.py`: `638a56e035856c600c9848b368be087cb706a61603a0790124968c95b8c69f3a`
  * `model/module.py`: `a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f`
* **One change:** in `kronos.py`, `sys.path.append("../")` and `from model.module import *` are replaced by the
  package-relative `from .module import *` (the original only imports when its parent folder is on `sys.path`, and
  `model` is too generic a top-level name). Nothing else differs; `module.py` is byte-identical.
* Sampling of *all* paths (the original `auto_regressive_inference` returns their mean) is implemented outside this
  folder, in `nexus_alpha/kronos/engine.py`, from the unmodified building blocks.
* Weights are not vendored: they are downloaded from the Hugging Face Hub at pinned revisions into the git-ignored
  `artifacts/kronos/` and hash-checked (`nexus_alpha/kronos/weights.py`).
