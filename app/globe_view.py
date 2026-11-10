"""Universe and world-macro globe for the Mission Control tab (globe.gl, MIT, pinned 2.46.2 via jsDelivr).

* Columns: where the forward-test universe's companies have their business address (SEC submissions endpoint,
  scripts/data/fetch_addresses.py), placed at the state, province or country's approximate centre and sized by the
  number of stocks. Addresses are current, not point in time: display context only, never a model input.
* Countries: one Global Macro Database variable (release 2026_09, latest actual year per country; projections dropped).
  Context only: macro blocks have no validated skill in the stock models (D-033), and the market read (E19) uses
  SPY and cross-asset features, not these.
* A pulse ring marks New York, where every forecast's 09:00 decision clock runs.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit.components.v1 as components

import ui as UI

GLOBE_JS = "https://cdn.jsdelivr.net/npm/globe.gl@2.46.2/dist/globe.gl.min.js"
EARTH_IMG = "https://cdn.jsdelivr.net/npm/three-globe@2.45.0/example/img/earth-dark.jpg"
COUNTRIES = "https://cdn.jsdelivr.net/gh/vasturiano/globe.gl@v2.46.2/example/datasets/ne_110m_admin_0_countries.geojson"

# approximate geographic centres (lat, lng); EDGAR state/country codes
US = {"AL": (32.8, -86.8), "AK": (64.2, -152.5), "AZ": (34.2, -111.7), "AR": (34.9, -92.4), "CA": (37.2, -119.5),
      "CO": (39.0, -105.5), "CT": (41.6, -72.7), "DC": (38.9, -77.0), "DE": (39.0, -75.5), "FL": (28.6, -82.4),
      "GA": (32.7, -83.4), "HI": (20.8, -156.3), "IA": (42.1, -93.5), "ID": (44.4, -114.6), "IL": (40.0, -89.2),
      "IN": (39.9, -86.3), "KS": (38.5, -98.4), "KY": (37.5, -85.3), "LA": (31.1, -92.0), "MA": (42.3, -71.8),
      "MD": (39.0, -76.8), "ME": (45.4, -69.2), "MI": (44.3, -85.4), "MN": (46.3, -94.3), "MO": (38.4, -92.5),
      "MS": (32.7, -89.7), "MT": (47.0, -109.6), "NC": (35.6, -79.4), "ND": (47.5, -100.5), "NE": (41.5, -99.8),
      "NH": (43.7, -71.6), "NJ": (40.2, -74.7), "NM": (34.4, -106.1), "NV": (39.3, -116.6), "NY": (42.9, -75.5),
      "OH": (40.3, -82.8), "OK": (35.6, -97.5), "OR": (43.9, -120.6), "PA": (40.9, -77.8), "PR": (18.2, -66.5),
      "RI": (41.7, -71.5), "SC": (33.9, -80.9), "SD": (44.4, -100.2), "TN": (35.9, -86.4), "TX": (31.5, -99.3),
      "UT": (39.3, -111.7), "VA": (37.5, -78.8), "VT": (44.1, -72.7), "WA": (47.4, -120.5), "WI": (44.6, -89.9),
      "WV": (38.6, -80.6), "WY": (43.0, -107.5)}
ABROAD = {"A0": (53.9, -116.6), "A1": (53.7, -127.6), "A6": (50.0, -85.0), "D0": (32.3, -64.8), "X0": (54.0, -2.0),
          "L2": (53.2, -8.0), "L3": (31.4, 35.0), "P7": (52.2, 5.5), "2M": (51.2, 10.4), "C1": (-38.4, -63.6),
          "D5": (-14.2, -51.9), "K8": (-2.5, 118.0), "G7": (56.0, 10.0), "G4": (35.1, 33.4), "H9": (64.0, 26.0),
          "E9": (19.3, -81.3), "N4": (49.8, 6.1), "M5": (36.5, 127.9), "R5": (-9.2, -75.0), "U0": (1.35, 103.82)}
NEW_YORK = (40.71, -74.01)

MACRO = {  # label -> (GMD variable, transform, colour stops low..high, unit)
    "Real GDP growth": ("rGDP", "growth", ["#FF6B6B", "#34506A", "#3FE0EE"], "%"),
    "Inflation": ("infl", "level", ["#3FE0EE", "#34506A", "#F2B544"], "%"),
    "Unemployment": ("unemp", "level", ["#3FE0EE", "#34506A", "#F2B544"], "%"),
    "Policy rate": ("cbrate", "level", ["#3FE0EE", "#34506A", "#F2B544"], "%"),
}


def universe_points(addresses: pd.DataFrame) -> tuple[list[dict], dict]:
    """Columns by location. Returns the points and a count of the stocks placed / not placed."""
    a = addresses.copy()
    code = a.state_or_country.fillna("")
    a["lat"] = [US.get(c, ABROAD.get(c, (np.nan, np.nan)))[0] for c in code]
    a["lng"] = [US.get(c, ABROAD.get(c, (np.nan, np.nan)))[1] for c in code]
    a["label"] = np.where(code.isin(list(US)), code + ", United States",
                          a.state_or_country_description.fillna(code))
    placed = a.dropna(subset=["lat", "lng"])
    pts = []
    for (lat, lng, lab), g in placed.groupby(["lat", "lng", "label"]):
        t = sorted(g.ticker)
        pts.append({"lat": float(lat), "lng": float(lng), "label": str(lab), "n": int(len(g)),
                    "tickers": ", ".join(t[:14]) + (f" +{len(t) - 14} more" if len(t) > 14 else "")})
    return sorted(pts, key=lambda p: -p["n"]), {"placed": int(len(placed)), "unplaced": int(len(a) - len(placed))}


def macro_by_country(label: str) -> tuple[dict, str]:
    """ISO3 -> [value, year] for the latest actual (non-projection) year of each country."""
    from nexus_alpha.macro import gmd
    var, how, _, _ = MACRO[label]
    raw = pd.read_csv(gmd._dir() / f"{var}_{gmd.RELEASE}.csv", usecols=lambda c: c in ("ISO3", "year", var, "forecast"),
                      low_memory=False)
    if "forecast" in raw:
        raw = raw[raw["forecast"].fillna(0) == 0]
    raw = raw.dropna(subset=[var]).sort_values(["ISO3", "year"])
    out = {}
    for iso, g in raw.groupby("ISO3"):
        if how == "growth":
            g = g[g[var] > 0]
            if len(g) < 2 or g.year.iloc[-1] - g.year.iloc[-2] != 1:
                continue
            out[iso] = [round(float(np.log(g[var].iloc[-1] / g[var].iloc[-2]) * 100), 2), int(g.year.iloc[-1])]
        else:
            out[iso] = [round(float(g[var].iloc[-1]), 2), int(g.year.iloc[-1])]
    years = pd.Series([v[1] for v in out.values()])
    return out, (f"{int(years.mode().iloc[0])}" if len(years) else "n/a")


def render_globe(points: list[dict], macro: dict, label: str, height: int = 540) -> None:
    _, _, stops, unit = MACRO[label]
    vals = np.array([v[0] for v in macro.values()]) if macro else np.array([0.0])
    lo, hi = (float(np.nanpercentile(vals, 10)), float(np.nanpercentile(vals, 90))) if len(vals) > 3 else (0.0, 1.0)
    if MACRO[label][1] == "growth":       # diverging around zero
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    nmax = max([p["n"] for p in points] or [1])
    doc = f"""<style>{UI._BASE_CSS}
#g {{ width: 100%; height: {height}px; }}
.lg {{ position: absolute; left: 12px; bottom: 12px; font-family: {UI.FONT_MONO}; font-size: 11px; color: {UI.MUTED};
  background: rgba(5,9,15,.72); border: 1px solid {UI.BORDER}; padding: 8px 10px; max-width: 46%; }}
.bar {{ width: 180px; height: 8px; margin: 6px 0 2px; background: linear-gradient(90deg, {stops[0]}, {stops[1]}, {stops[2]}); }}
.ends {{ display: flex; justify-content: space-between; width: 180px; }}
.tt {{ font-family: {UI.FONT_BODY}; font-size: 12px; background: rgba(12,22,34,.94); border: 1px solid {UI.ACCENT};
  padding: 6px 8px; color: {UI.TEXT}; max-width: 320px; }}
.tt b {{ font-family: {UI.FONT_HEAD}; }}
.err {{ position: absolute; inset: 0; display: grid; place-items: center; color: {UI.MUTED}; font-size: 13px; }}
</style>
<div style="position:relative"><div id="g"></div>
<div class="lg"><div style="color:{UI.TEXT}">{html.escape(label)} ({unit}), latest actual year</div>
<div class="bar"></div><div class="ends"><span>{lo:.1f}</span><span>{hi:.1f}</span></div>
<div style="margin-top:6px">Columns: forward-test companies by business address ({sum(p['n'] for p in points)} placed)</div></div></div>
<script src="{GLOBE_JS}"></script>
<script>
(() => {{
  const MACRO = {UI.json_script(macro)}, PTS = {UI.json_script(points)}, NMAX = {nmax};
  const LO = {lo}, HI = {hi}, STOPS = {UI.json_script(stops)};
  const el = document.getElementById('g');
  if (typeof Globe === 'undefined') {{
    el.innerHTML = '<div class="err">The globe library could not load (no internet connection?).</div>'; return; }}
  const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const mix = (a, b, t) => a.map((x, i) => Math.round(x + (b[i] - x) * t));
  const S = STOPS.map(hex);
  const colour = v => {{
    if (v === undefined || v === null || !isFinite(v)) return 'rgba(40,56,74,0.55)';
    let t = (v - LO) / ((HI - LO) || 1); t = Math.max(0, Math.min(1, t));
    const c = t < 0.5 ? mix(S[0], S[1], t * 2) : mix(S[1], S[2], (t - 0.5) * 2);
    return `rgba(${{c[0]}},${{c[1]}},${{c[2]}},0.92)`; }};
  const iso = f => (f.properties.ADM0_A3 && f.properties.ADM0_A3 !== '-99') ? f.properties.ADM0_A3 : f.properties.ISO_A3;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const G = new Globe(el)
    .width(el.clientWidth).height({height})
    .backgroundColor('rgba(0,0,0,0)')
    .globeImageUrl('{EARTH_IMG}')
    .showAtmosphere(true).atmosphereColor('{UI.ACCENT}').atmosphereAltitude(0.16)
    .pointsData(PTS).pointLat('lat').pointLng('lng')
    .pointAltitude(d => 0.012 + 0.16 * Math.sqrt(d.n / NMAX)).pointRadius(0.7)
    .pointColor(() => '{UI.ACCENT}')
    .pointLabel(d => `<div class="tt"><b>${{d.label}}</b><br>${{d.n}} stock${{d.n > 1 ? 's' : ''}}<br>${{d.tickers}}</div>`)
    .ringsData([{{lat: {NEW_YORK[0]}, lng: {NEW_YORK[1]}}}]).ringColor(() => t => `rgba(63,208,224,${{1 - t}})`)
    .ringMaxRadius(6).ringPropagationSpeed(2.2).ringRepeatPeriod(reduce ? 0 : 1300);
  G.pointOfView({{lat: 33, lng: -88, altitude: 2.05}});
  G.controls().autoRotate = !reduce; G.controls().autoRotateSpeed = 0.28;
  fetch('{COUNTRIES}').then(r => r.json()).then(geo => {{
    G.polygonsData(geo.features.filter(f => f.properties.ISO_A2 !== 'AQ'))
      .polygonAltitude(0.006).polygonsTransitionDuration(600)
      .polygonCapColor(f => colour((MACRO[iso(f)] || [])[0]))
      .polygonSideColor(() => 'rgba(63,208,224,0.06)')
      .polygonStrokeColor(() => 'rgba(63,208,224,0.22)')
      .polygonLabel(f => {{ const m = MACRO[iso(f)];
        return `<div class="tt"><b>${{f.properties.ADMIN}}</b><br>${{m ? m[0].toFixed(1) + ' {unit} (' + m[1] + ')' : 'no data'}}</div>`; }});
  }}).catch(() => {{}});
  addEventListener('resize', () => G.width(el.clientWidth));
}})();
</script>"""
    components.html(doc, height=height + 6)
