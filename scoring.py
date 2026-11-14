"""Relevance scoring: which PSI investigations matter most for a given crew habitat atmosphere."""
import math
import numpy as np
import pandas as pd

HABITATS = {  # name: (O2 fraction, pressure atm) - from the challenge statement
    "ISS / spacecraft cabin (21% O2, 14.7 psi)": (0.21, 1.0),
    "Moon base & pressurized rover (30% O2, 10.2 psi)": (0.30, 10.2 / 14.696),
    "Spacesuit (100% O2, 4.3 psi)": (1.00, 4.3 / 14.696),
    "Mars habitat (assumed = Moon base)": (0.30, 10.2 / 14.696),
}

# Hand-checked attributes (from PSI metadata + challenge text). AI extraction adds detail on top.
CURATED = {
    #          pillar(s)                         fuel phase   linked flight study (ground studies)
    "PSI-26":  (["prevention"],                  "solid",     None),   # BASS
    "PSI-25":  (["prevention", "suppression"],   "solid",     None),   # BASS-II
    "PSI-98":  (["prevention"],                  "solid",     None),   # Saffire-I
    "PSI-99":  (["prevention"],                  "solid",     None),   # Saffire-II
    "PSI-100": (["prevention"],                  "solid",     None),   # Saffire-III
    "PSI-102": (["detection"],                   "solid",     None),   # SAME
    "PSI-101": (["detection"],                   "solid",     None),   # SAME-R
    "PSI-47":  (["detection"],                   "dust",      None),   # DAFT
    "PSI-69":  (["suppression", "prevention"],   "liquid",    None),   # FLEX
    "PSI-68":  (["suppression", "prevention"],   "liquid",    None),   # FLEX-2
    "PSI-39":  (["prevention"],                  "liquid",    None),   # CFI
    "PSI-20":  (["prevention"],                  "gas (solid emulator)", None),  # ACME BRE
    "PSI-10":  (["fundamental"],                 "gas",       None),   # ACME Flame Design
    "PSI-21":  (["fundamental"],                 "gas",       None),   # ACME CLD
    "PSI-22":  (["fundamental"],                 "gas",       None),   # ACME E-FIELD
    "PSI-23":  (["fundamental"],                 "gas",       None),   # ACME s-Flame
    "PSI-159": (["fundamental", "prevention"],   "gas",       None),   # ACME CFI-G (highest O2)
    "PSI-107": (["fundamental", "detection"],    "gas",       None),   # SPICE (soot)
    "PSI-106": (["fundamental"],                 "gas",       None),   # SLICE
    "PSI-60":  (["fundamental", "detection"],    "gas",       "PSI-107"),
    "PSI-62":  (["prevention"],                  "solid",     "PSI-25"),
    "PSI-115": (["detection"],                   "solid",     "PSI-102"),
    "PSI-117": (["suppression"],                 "liquid",    "PSI-69"),
    "PSI-142": (["suppression"],                 "liquid",    "PSI-69"),
}
FUEL_WEIGHT = {"solid": 1.0, "gas (solid emulator)": 0.8, "liquid": 0.7, "gas": 0.5, "dust": 0.4}
WEIGHTS = {"pillar": 0.35, "envelope": 0.30, "fuel": 0.20, "data": 0.15}


def envelope_closeness(tests: pd.DataFrame, o2: float, p_atm: float) -> float:
    """exp(-d/0.15), d = distance to the nearest test in (O2 fraction, log10 pressure) space."""
    t = tests.dropna(subset=["o2_fraction", "pressure_atm"])
    if t.empty:
        return np.nan
    d = np.sqrt((t.o2_fraction - o2) ** 2 + (np.log10(t.pressure_atm) - math.log10(p_atm)) ** 2).min()
    return float(math.exp(-d / 0.15))


def score(inv: pd.DataFrame, tests: pd.DataFrame, o2: float, p_atm: float, weights=WEIGHTS) -> pd.DataFrame:
    rows = []
    n_tests = tests.groupby("psi_id").size()
    for _, r in inv.iterrows():
        pillars, phase, link = CURATED.get(r.id, (["fundamental"], "gas", None))
        P = 1.0 if set(pillars) & {"prevention", "detection", "suppression"} else 0.3
        if "fire safety" in str(r.sub_area).lower():
            P = max(P, 0.8)
        own = tests[tests.psi_id == r.id]
        E = envelope_closeness(own if not own.empty or not link else tests[tests.psi_id == link], o2, p_atm)
        e_known = not np.isnan(E)
        E = E if e_known else 0.5
        if link:  # ground study: conditions borrowed from its flight experiment, slightly discounted
            E *= 0.8
        F = FUEL_WEIGHT.get(phase, 0.5)
        D = math.log1p(n_tests.get(r.id, 0)) + math.log1p(r.n_publications or 0)
        rows.append(dict(id=r.id, name=r.acronym if isinstance(r.acronym, str) else r.title[:40],
                         pillars=", ".join(pillars), fuel_phase=phase, linked_to=link or "",
                         P=P, E=E, E_known=e_known, F=F, D_raw=D, n_tests=int(n_tests.get(r.id, 0))))
    s = pd.DataFrame(rows)
    s["D"] = s.D_raw / s.D_raw.max()
    for k, col in [("pillar", "P"), ("envelope", "E"), ("fuel", "F"), ("data", "D")]:
        s[f"pts_{k}"] = 100 * weights[k] * s[col]
    s["score"] = s[[c for c in s if c.startswith("pts_")]].sum(axis=1).round(1)
    return s.sort_values("score", ascending=False).reset_index(drop=True)
