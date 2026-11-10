# Third-party notices

What NEXUS Alpha takes from other projects, and how. Integration rules: `docs/DECISION_LOG.md` D-046 and
`docs/INTEGRATIONS.md`.

## Code or definitions used in the repository

| Project | Licence | What is used | Where |
|---|---|---|---|
| Microsoft Qlib (`microsoft/qlib`) | MIT, Copyright (c) Microsoft Corporation | The Alpha158 feature definitions (`qlib/contrib/data/loader.py`, `Alpha158DL`) and the semantics of its operators (`qlib/data/ops.py`, `qlib/data/_libs/rolling.pyx`), re-implemented; no Qlib source file is copied. pyqlib 0.9.7 runs only in a separate environment, to verify parity | `src/nexus_alpha/indicators/alpha158.py`, `scripts/verify/alpha158_qlib_parity.py` |
| Kronos (`shiyu-coder/Kronos`) | MIT, Copyright (c) 2025 ShiYu | Model code, vendored at a pinned commit | `src/nexus_alpha/kronos/vendor/` (its own `LICENSE` and `VENDORED.md`) |
| claude-trading-skills (`tradermonty/claude-trading-skills`) | MIT | The rally tracker, vendored unchanged | `src/nexus_alpha/rules/vendor/` |
| Magic UI (`magicuidesign/magicui`) | MIT, Copyright (c) Magic UI | The behaviour of NumberTicker, Marquee and AnimatedBeam, re-written in plain HTML/CSS/JS (no React, no source copied) | `app/ui.py` |
| Arwes (`arwes/arwes`) | MIT, Copyright (c) 2014-2024 Romel Perez | The framed-panel design language (corner brackets, thin luminous lines), recreated in CSS | `app/ui.py`, `.streamlit/config.toml` |

The MIT licence of each project above grants permission to use, copy, modify, merge, publish and distribute the
software, provided its copyright notice and permission notice are included; the full texts are in each project's
repository, and in `src/nexus_alpha/kronos/vendor/LICENSE` for the vendored Kronos files. The software is provided
"as is", without warranty of any kind.

## Loaded by the dashboard at run time (not redistributed)

| Resource | Licence | Pinned source |
|---|---|---|
| globe.gl (`vasturiano/globe.gl`) | MIT, Copyright (c) 2019 Vasco Asturiano | `cdn.jsdelivr.net/npm/globe.gl@2.46.2` |
| three-globe example texture `earth-dark.jpg` (`vasturiano/three-globe`) | MIT package, Copyright (c) 2019 Vasco Asturiano | `cdn.jsdelivr.net/npm/three-globe@2.45.0` |
| Natural Earth 1:110m countries (via the globe.gl repository) | public domain | `cdn.jsdelivr.net/gh/vasturiano/globe.gl@v2.46.2/example/datasets/` |
| IBM Plex Sans, Oxanium, JetBrains Mono | SIL Open Font Licence 1.1 | Google Fonts |

## Reference only (no code copied)

| Project | Licence | Use |
|---|---|---|
| World Monitor (`koala73/worldmonitor`) | AGPL-3.0 | Layout reference for the Mission Control tab (a situational-awareness board around a globe). AGPL code is not compatible with this repository's terms, so nothing is copied |
| awesome-dataviz (`hal9ai/awesome-dataviz`), awesome-webgl (`sjfricke/awesome-webgl`) | lists | Library selection: Plotly and deck.gl (pydeck) are already installed with the dashboard; globe.gl is the WebGL globe |

## Data

The Global Macro Database (Mueller, Xu, Lehbib and Chen 2025, NBER Working Paper 33714), release 2026_09, is used
under its Research Use Terms v1.1. Its files stay in `artifacts/` and are never committed; the dashboard cites it
wherever it shows it. Company addresses come from SEC's public submissions endpoint (display only).
