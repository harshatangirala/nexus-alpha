"""Point-in-time fundamental analysis from SEC XBRL company facts (E22; docs/V3_RESEARCH.md; decision D-039).

* `sec_facts`: download (owner's declared SEC User-Agent) and parse the needed us-gaap/dei concepts; facts filed on/after
  the seal are dropped at read time unless the sealed evaluation is open.
* `statements`: as-known-at-t statements: quarterly flows (3-month facts, Q4 = FY - 9M), trailing-twelve-month sums,
  latest balance-sheet instants, all using only facts filed on or before t.
* `ratios`: 31 ratios (valuation, profitability, growth, leverage, efficiency, quality), Piotroski F-score, Altman Z,
  a fixed-assumption DCF, plus the multiples shown in the dashboard.
"""
