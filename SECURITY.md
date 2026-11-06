# Security and data policy

## Secrets
* No credential is ever committed. API keys (`FRED_API_KEY`, `GEMINI_API_KEY`, Hugging Face tokens) live only in the
  owner's environment variables or in owner-held `.env` files outside the repository, loaded at run time by
  `nexus_alpha.secrets` (which never prints a value). `*.env` files are git-ignored.
* Before the repository was made public (2026-11-06) every git object in its history and the working tree was scanned
  for the key values themselves and for common token patterns: no hits.
* GitHub secret scanning and push protection are enabled on this repository.

## Data and model weights
Nothing licensed is published:
* price, macro or news data;
* chart caches;
* embeddings;
* article or book text;
* model weights.

All of it lives in `artifacts/` (git-ignored). The Global Macro Database is used under its Research Use Terms
(academic, non-commercial, no re-hosting). The sealed 2023+ evaluation period is opened only by the owner (`seal.unlock`).

## No trading connectivity
The code places no orders and holds no broker credentials. It is research and paper-trading decision support only.

## Reporting
Please report a security problem privately to the repository owner through GitHub (do not open a public issue for it).
