"""Gap finder, equipment insights, material lookup, habitat briefing and PDF report."""
import io, math, re
from datetime import date
import numpy as np
import pandas as pd

PSI_PER_ATM = 14.696

# ------------------------------------------------------------------ gap finder
CANDIDATES = [  # (label, O2 fraction, pressure psi) - habitat-relevant conditions worth testing
    ("ISS cabin", 0.21, 14.7),
    ("Moon base / rover (planned)", 0.30, 10.2),
    ("Moon base, lower-pressure option", 0.34, 8.0),
    ("Exploration atmosphere (between suit and cabin)", 0.32, 8.2),
    ("Spacesuit", 1.00, 4.3),
    ("Suit pre-breathe / airlock", 0.50, 6.0),
]
PHASE_GROUPS = {"Solid materials (fabrics, plastics)": ["solid", "solid (smoke)"],
                "Liquid droplets": ["liquid"], "Gas flames": ["gas"]}


def _dist(t, o2, p_psi):
    return np.sqrt((t.o2_fraction - o2) ** 2 + (np.log10(t.pressure_atm * PSI_PER_ATM) - math.log10(p_psi)) ** 2)


def coverage(tests, o2, p_psi, phases):
    t = tests[tests.fuel_phase.isin(phases)].dropna(subset=["o2_fraction", "pressure_atm"])
    if t.empty:
        return 0.0, None
    d = _dist(t, o2, p_psi)
    i = d.idxmin()
    return float(math.exp(-d.min() / 0.15)), t.loc[i]


def gap_table(tests):
    rows = []
    for label, o2, p in CANDIDATES:
        for group, phases in PHASE_GROUPS.items():
            c, near = coverage(tests, o2, p, phases)
            rows.append(dict(condition=label, o2_pct=round(o2 * 100), p_psi=p, fuel_type=group, coverage=round(c, 2),
                             nearest=(f"{near.investigation}: {near.o2_fraction * 100:.0f}% O2, "
                                      f"{near.pressure_atm * PSI_PER_ATM:.1f} psi") if near is not None else "none"))
    g = pd.DataFrame(rows)
    # priority: solids matter most for cabin fires; low coverage = big gap
    w = g.fuel_type.map({"Solid materials (fabrics, plastics)": 1.0, "Liquid droplets": 0.7, "Gas flames": 0.5})
    g["priority"] = ((1 - g.coverage) * w * 100).round(0)
    return g.sort_values("priority", ascending=False).reset_index(drop=True)


def coverage_grid(tests, phases, o2s, ps):
    t = tests[tests.fuel_phase.isin(phases)].dropna(subset=["o2_fraction", "pressure_atm"])
    if t.empty:
        return np.zeros((len(ps), len(o2s)))
    to2 = t.o2_fraction.values[None, None, :]
    tlp = np.log10(t.pressure_atm.values * PSI_PER_ATM)[None, None, :]
    O, P = np.meshgrid(o2s, np.log10(ps))
    d = np.sqrt((O[..., None] - to2) ** 2 + (P[..., None] - tlp) ** 2).min(axis=2)
    return np.exp(-d / 0.15)


# ------------------------------------------------------------------ materials
MATERIALS = {  # family: regex on fuel names
    "PMMA (acrylic / Plexiglas)": r"pmma|plexi|clear rod|black rod|acrylic|sphere",
    "Cotton-fiberglass fabric (SIBAL)": r"sibal|cotton",
    "Nomex (flame-resistant fabric)": r"nomex",
    "Silicone": r"silicone",
    "Kapton (polyimide film)": r"kapton",
    "Teflon (PTFE)": r"teflon",
    "Lampwick (cotton wick)": r"lampwick",
    "Pyrell (fire-retardant foam)": r"pyrell",
    "Dibutyl phthalate (plasticizer)": r"dibutyl|dbp",
}


def material_tests(tests, family):
    pat = MATERIALS[family]
    return tests[tests.fuel.astype(str).str.contains(pat, case=False, regex=True)]


def material_verdict(mt, o2, p_psi):
    if mt.empty:
        return "none", "No microgravity tests of this material in PSI."
    t = mt.dropna(subset=["o2_fraction", "pressure_atm"])
    max_o2 = t.o2_fraction.max() * 100
    measured = (t.o2_source == "measured").any()
    c = math.exp(-_dist(t, o2, p_psi).min() / 0.15) if len(t) else 0
    if c > 0.8:
        return "covered", f"Tested near these conditions ({len(mt)} tests)."
    note = "" if measured else " O2 was not recorded; cabin air assumed."
    return "gap", (f"Not covered. All {len(mt)} microgravity tests ran at ≤{max_o2:.0f}% O2 and about 14.7 psi.{note} "
                   f"Its behaviour at {o2 * 100:.0f}% O2 / {p_psi:.1f} psi is untested. (In FLEX droplet tests, more O2 meant faster "
                   f"burning and fewer flames going out on their own.)")


# ------------------------------------------------------------------ equipment insights
def suppression_summary(flex):
    f = flex.copy()
    f["diluent"] = np.select([f.he > 0, f.co2 > 0], ["Helium", "CO₂"], "Nitrogen (air-like)")
    f["o2_band"] = pd.cut(f.o2, [0, 0.15, 0.20, 0.25, 1], labels=["≤15%", "15–20%", "20–25%", ">25%"])
    return f.groupby(["o2_band", "diluent"], observed=True).agg(
        tests=("test", "size"), self_ext_pct=("self_extinguished", lambda s: round(100 * s.mean()))).reset_index()


SAFFIRE2 = pd.DataFrame([  # transcribed from PSI-99 experimental table (Saffire-II)
    ("Silicone 0.27 mm", "≈0 cm", "burned completely"),
    ("Silicone 0.61 mm", "≈0 cm", "7.6 cm"),
    ("Silicone 1.03 mm", "≈0 cm", "≈0 cm"),
    ("Cotton-fiberglass (SIBAL) 0.37 mm", "29 cm (whole sample, 2.1 mm/s)", "burned completely"),
    ("PMMA 0.85 mm + Nomex", "PMMA 5 cm, Nomex 0", "Nomex did not ignite"),
    ("PMMA (thick)", "≈10 cm damaged", "burned completely"),
], columns=["Sample", "Burn length in microgravity", "Burn length in 1 g (Earth)"])


# ------------------------------------------------------------------ briefing
def briefing(hab_name, o2, p_psi, ranked, tests, flex_pred=None):
    d = tests.dropna(subset=["o2_fraction", "pressure_atm"])
    near = d[(d.o2_fraction.sub(o2).abs() <= 0.03) & ((d.pressure_atm * PSI_PER_ATM / p_psi).between(0.8, 1.25))]
    solids_near = near.fuel_phase.str.startswith("solid").sum()
    top = ranked.head(3)
    lines = [f"**{hab_name}: {o2 * 100:.0f}% O₂ at {p_psi:.1f} psi.**",
             f"{len(near)} of {len(d):,} PSI tests ran within ±3% O₂ and ±20% pressure of this atmosphere"
             + (f", and {solids_near} of them burned solid materials." if len(near) else ", so there is no direct evidence.")]
    lines.append("Most relevant evidence: " + "; ".join(f"{r.name} ({r.id}, score {r.score:.0f})" for r in top.itertuples()) + ".")
    if flex_pred:
        lines.append(f"Predicted from FLEX: a methanol droplet flame burns at ≈{flex_pred['burn_rate']['mid']:.2f} mm²/s "
                     f"and goes out on its own {flex_pred['p_self_ext'] * 100:.0f}% of the time"
                     f" (vs 68% at ISS air).")
    if o2 > 0.24 and solids_near == 0:
        lines.append("**Biggest gap:** no microgravity test of a solid cabin material at this oxygen level.")
    return " ".join(lines)


# ------------------------------------------------------------------ PDF report
def pdf_report(hab_name, o2, p_psi, ranked, gaps, brief_text, flex_metrics):
    from fpdf import FPDF
    clean = lambda s: re.sub(r"\*\*", "", str(s)).replace("₂", "2").replace("≈", "~").replace("≤", "<=").replace("±", "+/-") \
        .replace("–", "-").replace("—", "-").replace("²", "^2").encode("latin-1", "replace").decode("latin-1")
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, clean(f"Flame in Freefall: habitat fire-safety brief"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, clean(f"{hab_name} | {o2 * 100:.0f}% O2 at {p_psi:.1f} psi | generated {date.today()}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 7, "Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10); pdf.multi_cell(0, 5, clean(brief_text), new_x="LMARGIN", new_y="NEXT"); pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 7, "Most relevant investigations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for r in ranked.head(10).itertuples():
        pdf.multi_cell(0, 5, clean(f"{r.score:5.0f}  {r.name} ({r.id}): {r.pillars}; {r.fuel_phase}; {r.n_tests} tests"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 7, "Biggest data gaps (recommended next tests)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for r in gaps.head(5).itertuples():
        pdf.multi_cell(0, 5, clean(f"- {r.fuel_type} at {r.o2_pct}% O2, {r.p_psi} psi ({r.condition}): coverage {r.coverage:.2f}; nearest {r.nearest}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, clean(
        f"Method: relevance = 35% fire-safety focus + 30% closeness of tested conditions + 20% fuel relevance + 15% data richness. "
        f"FLEX model cross-validated R^2 = {flex_metrics['burn_rate']['r2']:.2f} (burning rate). Where PSI tables omit O2/pressure, "
        f"ISS cabin air was assumed. Data: NASA Physical Sciences Informatics (psi.nasa.gov, CC0); cite investigations by DOI. "
        f"Text in this report is generated automatically from the data by the Flame in Freefall dashboard (built with AI assistance)."))
    return bytes(pdf.output())
