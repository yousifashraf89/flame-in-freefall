"""Moon-base fire-risk designer, atmosphere trade-off map and Earth oxygen-enriched settings."""
import math
import numpy as np
import pandas as pd

from features import MATERIALS, material_tests, _dist

PSI = 14.696
SUIT_PSI = 4.3  # spacesuit pressure (challenge statement)

# Cabin inventory -> closest material family tested in PSI
INVENTORY = {
    "Crew clothing & towels (cotton)": ["Cotton-fiberglass fabric (SIBAL)", "Lampwick (cotton wick)"],
    "Flame-resistant suits & bags (Nomex)": ["Nomex (flame-resistant fabric)"],
    "Windows, panels & displays (acrylic / PMMA)": ["PMMA (acrylic / Plexiglas)"],
    "Seals, gaskets & hoses (silicone)": ["Silicone"],
    "Wire insulation (Kapton)": ["Kapton (polyimide film)"],
    "Wire jackets & tubing (Teflon / PTFE)": ["Teflon (PTFE)"],
    "Fire-retardant foam & insulation": ["Pyrell (fire-retardant foam)"],
    "Soft plastics (plasticizers)": ["Dibutyl phthalate (plasticizer)"],
}
SMOKE_DATA = {"Kapton (polyimide film)", "Teflon (PTFE)", "Silicone", "Lampwick (cotton wick)",
              "Pyrell (fire-retardant foam)", "Dibutyl phthalate (plasticizer)"}  # SAME / SAME-R samples
NOTES = {
    "Silicone": "Saffire-II: thin silicone burned fully on Earth but barely in microgravity.",
    "Cotton-fiberglass fabric (SIBAL)": "Saffire-II: a 29 cm SIBAL sample burned its whole length in microgravity.",
    "Nomex (flame-resistant fabric)": "Saffire-II: Nomex did not ignite next to burning PMMA at ~22% O₂.",
    "PMMA (acrylic / Plexiglas)": "The most-tested solid in PSI (BASS, BASS-II, Saffire), but only at ≤22% O₂.",
}

# NFPA 99B "atmosphere of increased burning": O2 % > 23.45 / sqrt(total pressure in atm)
def increased_burning_limit_pct(p_psi):
    return 23.45 / math.sqrt(p_psi / PSI)


def atmosphere_flags(o2, p_psi):
    lim = increased_burning_limit_pct(p_psi)
    return dict(oea=o2 * 100 > 23.5, increased_burning=o2 * 100 > lim, limit_pct=lim,
                pp_o2_psi=o2 * p_psi, prebreathe_index=prebreathe_index(o2, p_psi))


def prebreathe_index(o2, p_psi, suit_psi=SUIT_PSI):
    """Relative decompression-stress proxy: cabin nitrogen partial pressure / suit pressure (assumes N2 is the diluent).
    Lower = less nitrogen to purge before a spacewalk. A comparison tool only, not a medical criterion."""
    return (1 - o2) * p_psi / suit_psi


def item_status(tests, family, o2, p_psi):
    mt = material_tests(tests, family)
    t = mt.dropna(subset=["o2_fraction", "pressure_atm"])
    if mt.empty:
        return dict(status="red", label="🔴 Never tested in microgravity", n=0, max_o2=None, coverage=0.0)
    c = float(math.exp(-_dist(t, o2, p_psi).min() / 0.15)) if len(t) else 0.0
    max_o2 = float(t.o2_fraction.max() * 100) if len(t) else None
    if (mt.fuel_phase == "solid (smoke)").all():
        return dict(status="orange", label="🟠 Only heated for smoke tests, never burned", n=len(mt), max_o2=max_o2,
                    coverage=0.0)
    if c > 0.8:
        return dict(status="green", label="🟢 Tested near these conditions", n=len(mt), max_o2=max_o2, coverage=c)
    return dict(status="yellow", label=f"🟡 Tested only at Earth-like air (≤{max_o2:.0f}% O₂)", n=len(mt), max_o2=max_o2,
                coverage=c)


def assess(tests, items, o2, p_psi, flex_model):
    rows = []
    for item in items:
        fams = INVENTORY[item]
        sts = [item_status(tests, f, o2, p_psi) for f in fams]
        best = max(sts, key=lambda s: s["coverage"])
        fam = fams[sts.index(best)]
        rows.append(dict(item=item, proxy=fam, **best, smoke_data=any(f in SMOKE_DATA for f in fams),
                         note=" ".join(NOTES[f] for f in fams if f in NOTES)))
    df = pd.DataFrame(rows)
    flags = atmosphere_flags(o2, p_psi)
    fx = flex_model.predict("Methanol", min(max(o2, 0.12), 0.40), min(max(p_psi, 7.0), 45.0) / PSI)
    readiness = float((df.status == "green").mean()) if len(df) else 0.0
    return df, flags, fx, readiness


def guidance(df, flags, fx, o2):
    det = sorted({f.split(" (")[0] for r in df.itertuples() for f in INVENTORY[r.item] if f in SMOKE_DATA})
    out = {
        "detection": ("Microgravity smoke data exists for " + ", ".join(det) +
                      ". Smoke particles grew 23–45% as they aged (SAME-R), so tune detectors for larger particles "
                      "far from the source." if det else
                      "No microgravity smoke data for the materials on board: detector response is unverified."),
        "suppression": (f"In FLEX, flames above 25% O₂ almost never went out on their own (8%)"
                        + (f"; at {o2:.0%} O₂ the model gives {fx['p_self_ext']:.0%}" if o2 <= 0.36 else
                           f"; {o2:.0%} O₂ is far beyond FLEX's tested range (≤34%), so expect even fewer")
                        + ". Plan active suppression and a way to lower O₂ quickly. Helium dilution put out the most "
                        "flames at 20–25% O₂." if o2 > 0.24 else
                        f"At {o2:.0%} O₂, FLEX droplet flames often went out on their own (model: {fx['p_self_ext']:.0%}). "
                        "Diluting with CO₂ or helium raises that further."),
        "tests": [f"{r.item}: burn {r.proxy.split(' (')[0]} at {o2:.0%} O₂ in microgravity" for r in df.itertuples()
                  if r.status != "green"],
    }
    return out


# ------------------------------------------------------------------ Earth: oxygen-enriched settings
EARTH = pd.DataFrame([
    # name, O2 low, O2 high (%), pressure low, high (psi), note
    ("Oxygen-enriched room (e.g. leaking oxygen supply)", 23.5, 30, 14.7, 14.7,
     "NFPA 99 and OSHA call air above 23.5% O₂ 'oxygen-enriched'. Upper bound shown is illustrative."),
    ("Multiplace hyperbaric chamber (air)", 21, 23.5, 29.4, 44.1,
     "Compressed air at 2–3 atm; chamber O₂ is kept at or below 23.5%."),
    ("Monoplace hyperbaric chamber (pure O₂)", 95, 100, 29.4, 44.1,
     "Patients breathe ~100% O₂ at 2–3 atm; chamber fires have killed patients."),
], columns=["setting", "o2_lo", "o2_hi", "p_lo", "p_hi", "note"])


# ------------------------------------------------------------------ figures
def tradeoff_figure(tests, coverage_grid, phases, o2_sel, p_sel, habitats):
    import plotly.graph_objects as go
    o2s, ps = np.linspace(0.18, 1.0, 83), np.geomspace(3.5, 16, 70)
    z = coverage_grid(tests, phases, o2s, ps)
    fig = go.Figure(go.Heatmap(x=o2s * 100, y=ps, z=z, zmin=0, zmax=1, colorbar=dict(title="fire data<br>coverage"),
                               colorscale=[[0, "#151622"], [0.4, "#1b2b85"], [0.75, "#2491ff"], [1, "#e7f434"]],
                               hovertemplate="%{x:.0f}% O₂, %{y:.1f} psi<br>data coverage %{z:.2f}<extra></extra>"))
    O, P = np.meshgrid(o2s, ps)
    fig.add_contour(x=o2s * 100, y=ps, z=prebreathe_index(O, P), showscale=False, hoverinfo="skip",
                    contours=dict(start=0.5, end=3.0, size=0.5, coloring="none", showlabels=True,
                                  labelfont=dict(color="#dfe1e2", size=11)),
                    line=dict(color="rgba(223,225,226,.55)", width=1, dash="dot"), name="pre-breathe index")
    pp = np.geomspace(3.5, 16, 60)
    fig.add_scatter(x=[increased_burning_limit_pct(p) for p in pp], y=pp, mode="lines", name="NFPA 99B: increased burning →",
                    line=dict(color="#ff5804", width=3))
    fig.add_scatter(x=np.clip(3.087 / pp * 100, 0, 100), y=pp, mode="lines", name="sea-level-equivalent oxygen (3.1 psi O₂)",
                    line=dict(color="#3DDC84", width=2, dash="dash"))
    for name, (ho2, hp) in habitats.items():
        fig.add_scatter(x=[ho2 * 100], y=[hp], mode="markers+text", text=[name],
                        **({} if ho2 > 0.9 else {"textposition": "top center"}),
                        marker=dict(symbol="star", size=16, color="#e7f434", line=dict(color="#151622", width=1)),
                        textfont=dict(color="#e7f434"), showlegend=False,
                        **({"textposition": "middle left"} if ho2 > 0.9 else {}))
    fig.add_scatter(x=[o2_sel * 100], y=[p_sel], mode="markers", name="your selection",
                    marker=dict(symbol="x", size=15, color="#ff5804", line=dict(width=2)))
    fig.update_layout(title="The habitat atmosphere trade-off: shorter pre-breathe ↔ more fire risk, less data",
                      xaxis=dict(title="Oxygen (%)", range=[18, 104]),
                      yaxis=dict(title="Cabin pressure (psi, log scale)", type="log"),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def earth_figure(tests):
    import plotly.graph_objects as go
    d = tests.dropna(subset=["o2_fraction", "pressure_atm"])
    fig = go.Figure()
    colors = {"solid": "#FF5D5D", "solid (smoke)": "#e7f434", "liquid": "#3DDC84", "gas": "#2491ff"}
    for ph, g in d.groupby("fuel_phase"):
        fig.add_scatter(x=g.o2_fraction * 100, y=g.pressure_atm * PSI, mode="markers", name=f"PSI tests: {ph}",
                        marker=dict(size=6, color=colors.get(ph, "#888"), opacity=.55))
    for r in EARTH.itertuples():
        fig.add_shape(type="rect", x0=r.o2_lo, x1=r.o2_hi, y0=r.p_lo * 0.97, y1=r.p_hi * 1.03,
                      line=dict(color="#ff5804", width=2), fillcolor="rgba(255,88,4,.18)")
        fig.add_annotation(x=min((r.o2_lo + r.o2_hi) / 2, 88), y=math.log10(r.p_hi * 1.12), text=r.setting, showarrow=False,
                           font=dict(color="#ffb38a", size=12))
    fig.update_layout(title="Oxygen-rich places on Earth vs where NASA's microgravity tests ran",
                      xaxis=dict(title="Oxygen (%)", range=[5, 104]),
                      yaxis=dict(title="Pressure (psi, log scale)", type="log", range=[0.55, 1.85],
                                 tickvals=[4, 5, 7, 10, 14.7, 20, 30, 45, 70], ticktext=["4", "5", "7", "10", "14.7", "20", "30", "45", "70"]),
                      legend=dict(orientation="h", y=-0.2))
    return fig
