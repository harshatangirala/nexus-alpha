"""Plain-language project overview PDF with flow-chart diagrams (about 1,000 words).

    python scripts/report/make_overview_pdf.py
Output: reports/NEXUS_Alpha_Overview.pdf
"""
from __future__ import annotations

from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "NEXUS_Alpha_Overview.pdf"
W = A4[0] - 36 * mm
INK, MUTED = colors.HexColor("#1f2933"), colors.HexColor("#52606d")
BLUE, GREEN, AMBER, GREY = (colors.HexColor(c) for c in ("#dbeafe", "#dcfce7", "#fef3c7", "#f1f5f9"))
EDGE = colors.HexColor("#64748b")


def box(d, x, y, w, h, title, lines=(), fill=BLUE, size=7.2):
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=EDGE, strokeWidth=0.7))
    d.add(String(x + w / 2, y + h - 11, title, fontName="Helvetica-Bold", fontSize=8, fillColor=INK, textAnchor="middle"))
    for i, ln in enumerate(lines):
        d.add(String(x + w / 2, y + h - 22 - i * (size + 2), ln, fontName="Helvetica", fontSize=size, fillColor=INK,
                     textAnchor="middle"))


def arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=EDGE, strokeWidth=1))
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    p = [x2, y2, x2 - 6 * math.cos(a - 0.4), y2 - 6 * math.sin(a - 0.4), x2 - 6 * math.cos(a + 0.4), y2 - 6 * math.sin(a + 0.4)]
    d.add(Polygon(p, fillColor=EDGE, strokeColor=EDGE))


def caption(d, text):
    d.add(String(0, 2, text, fontName="Helvetica-Oblique", fontSize=7.5, fillColor=MUTED))


def pipeline():
    d = Drawing(W, 175)
    bw, bh, gap = 150, 62, (W - 3 * 150) / 2
    top = [("1. Data (free)", ["500 US stocks 2001-2022", "news, SEC filings, SPY/ETFs,", "macro (GMD, FRED)"], GREY),
           ("2. Honesty checks", ["only information known", "at decision time;", "2023+ locked (sealed)"], AMBER),
           ("3. Features", ["80 indicators; 3 chart", "images (week/day/hour);", "news + filing text"], BLUE)]
    bot = [("6. Outputs", ["probabilities per horizon,", "ranges, plan, context,", "logged every night"], GREEN),
           ("5. Risk engine", ["trade only with proven skill;", "entry, stop, size;", "discipline checks"], AMBER),
           ("4. Models", ["boosted trees, logistic,", "fusion network, GRU;", "blended + calibrated"], BLUE)]
    for i, (t, ls, f) in enumerate(top):
        box(d, i * (bw + gap), 100, bw, bh, t, ls, f)
        if i < 2:
            arrow(d, i * (bw + gap) + bw, 131, (i + 1) * (bw + gap), 131)
    arrow(d, 2 * (bw + gap) + bw / 2, 100, 2 * (bw + gap) + bw / 2, 82)
    for i, (t, ls, f) in enumerate(bot):
        box(d, i * (bw + gap), 18, bw, bh, t, ls, f)
        if i < 2:
            arrow(d, (i + 1) * (bw + gap), 49, i * (bw + gap) + bw, 49)
    caption(d, "Figure 1. The whole system, from raw data to what you see.")
    return d


def desk():
    d = Drawing(W, 160)
    box(d, 0, 90, 120, 58, "You give", ["ticker + trade type", "3 screenshots (W/D/H)", "1-10 news / macro texts"], GREY)
    box(d, 160, 110, 160, 40, "Real bars fetched", ["the model reads real prices,", "not screenshot pixels"], BLUE)
    box(d, 160, 60, 160, 40, "Screenshot check", ["legend values vs real bars", "(Gemini read them 228/228 exact)"], AMBER)
    box(d, 160, 16, 160, 40, "Text analysis", ["news branch of the model +", "optional Gemini read (logged)"], BLUE)
    for y in (130, 80, 36):
        arrow(d, 120, 119, 160, y)
    box(d, 360, 30, W - 360, 120, "You get", ["1. horizon mapping", "2. v1.1 + v2 probabilities,", "    return range, skill flag",
                                            "3. trade plan (or NO TRADE)", "4. SPY 20-day lean, macro,", "    regime rules",
                                            "5. screenshot check  6. Gemini read", "7. discipline gate + sizing"], GREEN, size=7)
    for y in (130, 80, 36):
        arrow(d, 320, y, 360, 90)
    caption(d, "Figure 2. What happens on the Trade Desk when you upload screenshots and news.")
    return d


def ladder():
    d = Drawing(W, 95)
    steps = [("Train", "2003-15", GREY), ("Inner", "2016 stop", GREY), ("Validation", "2017-18 choose", BLUE),
             ("Calibrate", "2019", BLUE), ("Test", "2020-22 check", AMBER), ("Sealed", "2023+ once", AMBER),
             ("Forward", "nightly, live", GREEN)]
    bw = (W - 6 * 10) / 7
    for i, (t, s, f) in enumerate(steps):
        box(d, i * (bw + 10), 40, bw, 48, t, [s], f)
        if i < 6:
            arrow(d, i * (bw + 10) + bw, 64, (i + 1) * (bw + 10), 64)
    d.add(String(0, 22, "Choices are made only on validation. Test is a reused check. Sealed and forward data are the "
                        "real exams.", fontName="Helvetica", fontSize=7.5, fillColor=INK))
    caption(d, "Figure 3. How the model is kept honest over time.")
    return d


def fishbone():
    d = Drawing(W, 190)
    y0 = 100
    d.add(Line(20, y0, W - 120, y0, strokeColor=INK, strokeWidth=1.6))
    box(d, W - 118, y0 - 22, 118, 44, "Effect", ["no proven, tradeable", "edge yet"], AMBER)
    ribs_top = [("Data", "survivor universe; news on 46%"), ("Methods", "direction target; 1-year calibration"),
                ("Measurement", "fixed: DSR, warm-up, 6M/1Y claims")]
    ribs_bot = [("Infrastructure", "fixed: BLAS, Yahoo hangs, calendar"), ("People/process", "post-result decisions disclosed"),
                ("Market", "near random walk; costs; regimes")]
    xs = [70, 200, 330]
    for (t, s), x in zip(ribs_top, xs):
        d.add(Line(x - 40, y0 + 62, x, y0, strokeColor=EDGE, strokeWidth=1))
        d.add(String(x - 44, y0 + 72, t, fontName="Helvetica-Bold", fontSize=8, fillColor=INK))
        d.add(String(x - 44, y0 + 63, s, fontName="Helvetica", fontSize=6.6, fillColor=INK))
    for (t, s), x in zip(ribs_bot, xs):
        d.add(Line(x - 40, y0 - 62, x, y0, strokeColor=EDGE, strokeWidth=1))
        d.add(String(x - 44, y0 - 72, t, fontName="Helvetica-Bold", fontSize=8, fillColor=INK))
        d.add(String(x - 44, y0 - 81, s, fontName="Helvetica", fontSize=6.6, fillColor=INK))
    caption(d, "Figure 4. Fishbone (cause-and-effect) diagram from the audit.")
    return d


def main() -> None:
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=19, textColor=INK, spaceAfter=4)
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, textColor=INK, spaceBefore=8, spaceAfter=3)
    P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9.6, leading=13.2, textColor=INK, spaceAfter=4)
    B = ParagraphStyle("b", parent=P, leftIndent=10, bulletIndent=0, spaceAfter=1.5)
    sub = ParagraphStyle("s", parent=P, textColor=MUTED, fontSize=9)
    bl = lambda t: Paragraph(t, B, bulletText="-")
    story = [
        Paragraph("NEXUS Alpha: the project in plain words", H1),
        Paragraph("Overview for the owner, 7 November 2026. Code and full details: github.com/harshatangirala/nexus-alpha", sub),
        Paragraph("What it is", H2),
        Paragraph("NEXUS Alpha is a research system that looks at a US stock the way a trader does: three charts (weekly, "
                  "daily, hourly), technical indicators, news, company filings, the company's financial statements and "
                  "the wider market. It then estimates "
                  "the chances that the stock goes up, sideways or down over a chosen period, from 1 day to 1 year, and "
                  "turns that into a careful, rules-based trade plan. It is built to be honest: every result is checked "
                  "against data the model never saw, and nothing is ever sent to a broker.", P),
        Paragraph("How it works", H2),
        pipeline(),
        Paragraph("Data comes only from free sources. Strict checks make sure the model only uses information that "
                  "existed at the moment of the decision (09:00 New York, before the market opens). The years 2023 onward "
                  "are locked away for one final exam. Features are built from prices, chart images, text and macro data; "
                  "several model types are blended, calibrated, and passed to a risk engine that refuses to trade unless "
                  "skill has been proven.", P),
        KeepTogether([Paragraph("What you give and what you get", H2), desk()]),
        Paragraph("Your screenshots are used to confirm the ticker and to check that the prices you see match the real "
                  "data. The model itself reads clean, real price bars, because that is what it was trained on. Texts "
                  "feed the news branch, and Gemini (optional, free tier) adds a written read that is logged so it can "
                  "be scored later.", P),
    ]
    rows = [["Part", "What the tests found"],
            ["Technical indicators", "The strongest single input"],
            ["Chart images (computer vision)", "Small extra value"],
            ["News and filings (NLP)", "Weak; news exists for 46% of samples"],
            ["Macro and S&P 500 data", "No gain per stock; a useful SPY 20-day lean (AUROC 0.72 / 0.68)"],
            ["Fundamentals (31 ratios, SEC)", "Help ranking within each industry (E23 kept); in the forward test"],
            ["Up / sideways / down", "Better than the base rate at 1-20 days; direction itself is near a coin flip"],
            ["Ranking stocks against each other", "Real but small skill (rank IC 0.02-0.05); not yet shown to beat costs"]]
    t = Table(rows, colWidths=[W * 0.36, W * 0.64])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.6), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.6),
                           ("BACKGROUND", (0, 0), (-1, 0), BLUE), ("GRID", (0, 0), (-1, -1), 0.4, EDGE),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story += [KeepTogether([Paragraph("What each part contributes (measured, not assumed)", H2), t]), Spacer(1, 6),
              Paragraph("In short: the system is good at judging how much a stock is likely to move, a little good at "
                        "telling which stocks will do better than others, and not yet able to say which way a single "
                        "stock will go. That is normal for markets, and it is why the project measures everything "
                        "instead of trusting backtests.", P),
              Paragraph("Fundamental analysis", H2),
              Paragraph("Every company's financial statements come from the SEC's free XBRL data (from 2009), exactly "
                        "as they were known on each day: a restated number only counts once it was filed. From them the "
                        "system computes 31 ratios (valuation such as earnings and free-cash-flow yield, margins, returns "
                        "on capital, growth, debt and liquidity, efficiency, and quality scores: Piotroski, Altman Z and "
                        "the earnings surprise), plus P/E, EV/EBITDA, P/S, P/B, P/FCF and a discounted cash flow you can "
                        "adjust. The Trade Desk shows the three statements, every ratio with the stock's rank in the "
                        "whole market and within its industry, and the DCF.", P),
              Paragraph("A strict data audit came first. It caught market values that were wrong by a split factor, by "
                        "the ADR ratio of foreign companies, by share class, and by filings typed in thousands. All are "
                        "now guarded, and the ten largest companies match their public values. Two pre-registered tests "
                        "followed. Ratios ranked across the whole market did not pass the rule. Ranked within each "
                        "industry, they did: they improved the ranking of stocks, mostly over months. That model "
                        "(v3_fund) is frozen and joins the nightly forward test. It becomes a shown signal only if it "
                        "beats the current ranker there.", P),
              Paragraph("How we keep it honest", H2), ladder(),
              Paragraph("Every evening the frozen v1.1, v2 and v3_fund models forecast about 500 stocks, and the frozen "
                        "SPY market read forecasts the market. The forecasts are hash-chained and pushed to the public "
                        "repository before the market opens, so nobody can change them after the fact. A one-time sealed "
                        "test of 2023 onward is ready; only you can open it.", P),
              KeepTogether([Paragraph("The audit (SWOT, fishbone, Six Sigma)", H2), fishbone()]),
              Paragraph("Two audits were run back to back, following Six Sigma's Define-Measure-Analyze-Improve-Control "
                        "steps. Six problems were found and fixed:", P),
              bl("The forward scorer's labels now match training 99.2% of the time (was 98.4%)."),
              bl("A broken return column in the v2 data was repaired."),
              bl("The decision-time rule for live forecasts beyond the calendar was corrected."),
              bl("The dashboard no longer claims skill at 6 months and 1 year."),
              bl("The calendar was extended to 2030."),
              bl("The backtest is now reported with its deflated Sharpe ratio (about 0, so no proven profit)."),
              Paragraph("A new test (E21) tried three dedicated stock rankers; none beat the existing model, which now has "
                        "its rankings scored every night. A third and fourth audit (fundamentals and the whole project) "
                        "found 15 more issues and fixed 12 (the other 3 are documented data limits), including a Yahoo "
                        "outage guard for the nightly run and a test that checks the nightly script itself. Control charts and the full test suite (109 tests, all passing) run "
                        "every night.", P),
              Paragraph("Is it complete without paid subscriptions?", H2),
              Paragraph("Yes. Everything runs on free data and a normal CPU. Gemini, QVeris and a GPU are optional extras.", P),
              Paragraph("What is left", H2),
              bl("<b>You:</b> sign in to QVeris (/mcp) if you want its data."),
              bl("<b>You:</b> open the sealed 2023+ test once, when ready (commands in docs/SEALED_EVALUATION_PLAN.md, 7b; "
                 "about 3-5 hours on CPU)."),
              bl("<b>Optional:</b> add two retry times to the nightly task (one command in "
                 "docs/AUDIT_3_4_2026-11-07.md, section 5)."),
              bl("<b>Time:</b> let the nightly forward test run. About 20 sessions before the control charts mean "
                 "anything, and about 6 months (120 sessions) before v3_fund's fundamentals ranking can be confirmed "
                 "or rejected."),
              bl("<b>Next research (pre-registered):</b> a live news feed variant, a longer calibration window, an "
                 "earnings-drift test, and the stock setups from the trading-skill libraries."),
              bl("<b>Rule:</b> no real trading decisions until the forward test and the sealed test both show skill "
                 "after costs.")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                            bottomMargin=15 * mm, title="NEXUS Alpha overview", author="NEXUS Alpha project")
    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
