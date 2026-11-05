"""Macro and market-context data for NEXUS v2 (docs/V2_ARCHITECTURE.md).

Every series is stored with the time it became available, and every feature is joined as-of the decision time
(09:00 New York on session t+1). Sources: Yahoo market series, index/sector/macro-proxy ETF bars from TriFrame's minute
source, the Global Macro Database (annual, publication-lagged), FRED/ALFRED vintages (when FRED_API_KEY is set), DBnomics.
Development reads stop at the seal (2023-01-03).
"""
