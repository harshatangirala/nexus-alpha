# QVeris MCP: connection status and how NEXUS uses it

QVeris is a hosted MCP marketplace that routes tool calls to data providers. Discovery is free; calls cost credits.
Docs: `@qverisai/mcp` (seven tools: `discover`, `inspect`, `probe`, `call`, `usage_history`, `credits_ledger`,
`lookup_execution`).

## Connections (Claude Code, user scope)

| Server | Endpoint | Status (2026-11-06) | Tools |
|---|---|---|---|
| `qveris` | `https://mcp.qveris.ai/mcp` (OAuth discovery) | **needs the owner's browser sign-in**: run `/mcp` in an interactive Claude Code session and choose qveris. No API key is created or stored | all seven |
| `qveris-public` | `https://mcp.qveris.ai/mcp/readonly` (no credentials) | connected | `discover`, `inspect` only (no prices, no calls) |

The public endpoint was verified with plain JSON-RPC (initialize, tools/list, discover, inspect). Short keyword queries
work ("news", "earnings", "options"); long sentences returned nothing.

## Providers found for NEXUS's data gaps (public discovery, 2026-11-06)

| Gap in NEXUS | Candidate QVeris tool | Notes |
|---|---|---|
| No live news feed for the forward test | `alphavantage.news_sentiment.retrieve.v1.7aca3c4a` (tickers, `time_from`/`time_to`, up to 1000 items); `finnhub.news.execute.v1` | Point-in-time windows are possible. A different source from the training corpus, so it could only be used for frozen models as a **new, pre-registered** forward variant |
| Earnings dates and surprises (post-earnings drift tests) | `finnhub.calendar.earnings.retrieve.v1.1552775d`, `finnhub.stock.earnings.retrieve.v1`, `alphavantage.earnings_estimates.retrieve.v1.*` | Needed for a PEAD test (D-034 candidate) |
| Live 8-K filings | `financialmodelingprep.stable.secfilings8k.retrieve.v1.1cbaa60c` | The Trade Desk already reads EDGAR directly when `SEC_USER_AGENT` is set |
| Intraday and live prices | `eodhd.intraday.retrieve.v1.82292199`, `eodhd.live_prices.retrieve.v1.178541d4` | Yahoo already serves the live path; EODHD could be a fallback when Yahoo blocks |

## Policy

* **Credits:** discover before calling; `probe` (quote) before any paid call; check `usage_history` / `credits_ledger`
  after.
* **No training data from QVeris.** A pay-per-call marketplace is not a reproducible historical source; models are trained
  on the documented local sources only (`docs/V2_ARCHITECTURE.md` section 3).
* **Forward test.** Adding a QVeris feed to a frozen model's live inputs would change what is being tested. It would be a
  separate, pre-registered variant with its own frozen record.
