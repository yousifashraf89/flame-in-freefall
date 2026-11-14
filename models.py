"""
Predictive models trained on PSI microgravity data.
  1. FLEX droplet fires (PSI-69): burning rate (median + 10-90% band), burn time, P(self-extinction)
  (A CFI-G ignition model was tried and dropped: it did not beat the base rate.)
Every prediction carries an extrapolation check: how far the query is from the nearest real test.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"


def raw_path(name: str) -> Path:
    """Find a raw PSI table wherever it was uploaded in the repo (data/raw, data, repo root, any subfolder)."""
    for p in (RAW / name, ROOT / "data" / name, ROOT / name):
        if p.exists():
            return p
    hits = sorted(ROOT.rglob(name))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{name} not found. Upload it to data/raw/ in the repo.")


def o2_monotone_clf(n_features, o2_index):
    """Physics-informed classifier: P(flame goes out on its own) may only fall as oxygen rises."""
    mono = [0] * n_features
    mono[o2_index] = -1
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, monotonic_cst=mono, random_state=0)


def _read(name):
    d = pd.read_csv(raw_path(name), encoding="latin-1")
    d.columns = [c.replace("ï»¿", "").strip() for c in d.columns]
    return d


# ---------------------------------------------------------------- FLEX
FLEX_FEATURES = ["is_heptane", "o2", "p_atm", "co2", "he", "d0_mm"]


def flex_data() -> pd.DataFrame:
    d = _read("PSI-69__FLEX.csv")
    n = lambda c: pd.to_numeric(d[c], errors="coerce")
    out = pd.DataFrame({
        "test": d["FLEX Identifier"], "fuel": d["Fuel"].str.strip(),
        "is_heptane": (d["Fuel"].str.strip() == "Heptane").astype(int),
        "o2": n("O initial ambient composition; mole fraction"),
        "p_atm": n("Ambient pressure; mmHg") / 760.0,
        "co2": n("CO initial ambient composition; mole fraction").fillna(0),
        "he": n("He initial ambient composition; mole fraction").fillna(0),
        "d0_mm": n("Droplet initial diameter; mm"),
        "burn_rate": n("Burning rate; mm"),           # burning-rate constant K (mm^2/s)
        "burn_time": n("Burn time; s"),
        "ext_d_mm": n("Visible flame extinction diameter; mm"),
        "end": d["Test end"].str.strip(),
    })
    out = out[out.p_atm.between(0.1, 5) & out.o2.between(0.05, 1)]
    out["self_extinguished"] = (out.end == "Extinction").astype(int)
    return out.reset_index(drop=True)


class FlexModel:
    def __init__(self):
        self.df = flex_data()
        self.reg, self.metrics = {}, {}
        for target in ["burn_rate", "burn_time"]:
            t = self.df.dropna(subset=FLEX_FEATURES + [target])
            X, y = t[FLEX_FEATURES].values, t[target].values
            self.reg[target] = {q: GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200, max_depth=3,
                                                             learning_rate=0.05, random_state=0).fit(X, y)
                                for q in (0.1, 0.5, 0.9)}
            med = GradientBoostingRegressor(loss="absolute_error", n_estimators=200, max_depth=3, learning_rate=0.05,
                                            random_state=0)
            pred = cross_val_predict(med, X, y, cv=KFold(5, shuffle=True, random_state=0))
            self.metrics[target] = dict(n=len(y), mae=float(np.mean(np.abs(pred - y))), median=float(np.median(y)),
                                        r2=float(1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)))
        t = self.df.dropna(subset=FLEX_FEATURES)
        X, y = t[FLEX_FEATURES].values, t.self_extinguished.values
        clf = o2_monotone_clf(X.shape[1], FLEX_FEATURES.index("o2"))
        prob = cross_val_predict(clf, X, y, cv=KFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        self.metrics["self_extinguished"] = dict(n=len(y), accuracy=float(np.mean((prob > 0.5) == y)),
                                                 base_rate=float(y.mean()))
        self.clf = clf.fit(X, y)
        self._train = t[FLEX_FEATURES].values
        self._scale = np.nanstd(self._train, axis=0) + 1e-9

    def predict(self, fuel, o2, p_atm, co2=0.0, he=0.0, d0_mm=3.0) -> dict:
        x = np.array([[int(fuel == "Heptane"), o2, p_atm, co2, he, d0_mm]])
        res = {t: {q: float(m.predict(x)[0]) for q, m in qs.items()} for t, qs in self.reg.items()}
        for t in res:  # keep the band ordered
            lo, mid, hi = sorted(res[t].values())
            res[t] = {"low": lo, "mid": mid, "high": hi}
        res["p_self_ext"] = float(self.clf.predict_proba(x)[0, 1])
        res["distance"] = nearest_distance(self._train, x[0], self._scale, cols=[1, 2, 3, 4])
        res["n_similar"] = int(np.sum((np.abs(self._train[:, 1] - o2) < 0.03) &
                                      (np.abs(np.log(self._train[:, 2] / p_atm)) < 0.25) &
                                      (self._train[:, 0] == int(fuel == "Heptane"))))
        return res


def nearest_distance(train, x, scale, cols):
    """Standardized distance to the nearest training point, using only the condition columns."""
    t = train[:, cols]
    ok = ~np.isnan(t).any(axis=1)
    return float(np.min(np.sqrt((((t[ok] - x[cols]) / scale[cols]) ** 2).sum(axis=1))))


def reliability(distance):
    if distance < 0.5:
        return "high", "✅ Close to real tests"
    if distance < 1.5:
        return "medium", "⚠️ Between tested conditions: treat as an estimate"
    return "low", "⛔ Far from any test: extrapolation, do not rely on it"


# ---------------------------------------------------------------- CFI-G gas flames (PSI-159)
def cfig_flames() -> pd.DataFrame:
    rows = []
    for f, flame, o2col in [("PSI-159__CFI-G_Normal_Flames.csv", "normal", "XO2 before test"),
                            ("PSI-159__CFI-G_Inverse_Flames.csv", "inverse", "XO2")]:
        d = _read(f)
        term = [c for c in d.columns if c.lower().startswith("self")][0]
        hc = [c for c in d.columns if c.startswith("mHC") or c.startswith("HC burnrate")][0]
        fuel_raw = d["Fuel"].astype(str).str.strip().str.lower()
        frac = fuel_raw.str.extract(r"(\d+)\s*%")[0].astype(float).div(100).fillna(1.0)
        base = fuel_raw.str.replace(r"\d+\s*%", "", regex=True).str.replace("n-", "", regex=False).str.strip()
        rows.append(pd.DataFrame({
            "fuel": base, "fuel_frac": frac, "inverse": int(flame == "inverse"),
            "o2": pd.to_numeric(d[o2col], errors="coerce"),
            "p_atm": pd.to_numeric(d["approx p (bar)"], errors="coerce") / 1.01325,
            "fuel_flow": pd.to_numeric(d[hc], errors="coerce"),
            "term": d[term].astype(str).str.strip(), "ign": d["Hot ignition"].astype(str).str.strip().str.lower()}))
    g = pd.concat(rows, ignore_index=True)
    # normal flames only: there XO2 is the surrounding atmosphere (in inverse flames it is the burner gas)
    g = g[(g.ign == "yes") & g.term.isin(["SE", "FT"]) & (g.inverse == 0)].dropna(subset=["o2", "p_atm"]).copy()
    g["self_ext"] = (g.term == "SE").astype(int)
    return g.reset_index(drop=True)


class GasFlameModel:
    """Chance that a burning gas flame (e.g. a small leak) puts itself out before the fuel is shut off.
    Trained on ACME CFI-G normal spherical flames: ambient O2 18-40%, 0.5-3 bar, ethane/propane/butane."""
    def __init__(self):
        g = cfig_flames()
        self.fuels = sorted(g.fuel.value_counts().loc[lambda s: s >= 10].index)
        g = g[g.fuel.isin(self.fuels)]
        self.df = g
        X, y = self._X(g.fuel, g.fuel_frac, g.o2, g.p_atm, g.fuel_flow.fillna(g.fuel_flow.median())), g.self_ext.values
        clf = o2_monotone_clf(X.shape[1], len(self.fuels) + 1)
        prob = cross_val_predict(clf, X, y, cv=KFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        self.metrics = dict(n=len(y), accuracy=float(np.mean((prob > 0.5) == y)), base_rate=float(max(y.mean(), 1 - y.mean())))
        self.clf = clf.fit(X, y)
        self._train, self._scale = X, X.std(axis=0) + 1e-9
        self.flow_median = float(g.fuel_flow.median())

    def _X(self, fuel, frac, o2, p, flow):
        fuel = np.asarray(fuel)
        cols = [fuel == f for f in self.fuels] + [np.asarray(frac, float), np.asarray(o2, float),
                                                  np.log10(np.asarray(p, float)), np.asarray(flow, float)]
        return np.column_stack(cols).astype(float)

    def predict(self, fuel, frac, o2, p_atm, flow=None):
        x = self._X([fuel], [frac], [o2], [p_atm], [flow if flow is not None else self.flow_median])
        k = len(self.fuels)
        return dict(p_self_ext=float(self.clf.predict_proba(x)[0, 1]),
                    distance=nearest_distance(self._train, x[0], self._scale, cols=[k + 1, k + 2]))


# ---------------------------------------------------------------- SAME-R smoke (PSI-101)
class SmokeModel:
    """How much smoke particles grow as smoke ages (Rt/R0), by material, air flow and aging time."""
    def __init__(self):
        s = _read("PSI-101__SAME-R.csv").dropna(subset=["Rt/R0"])
        s["age"] = pd.to_numeric(s["Net aging time (sec)"], errors="coerce")
        s["age"] = s.age.fillna(s.age.median())
        s["v"] = pd.to_numeric(s["Inlet velocity (cm/s)"], errors="coerce").fillna(0)
        self.materials = sorted(s["Sample Type"].unique())
        self.df = s
        X, y = self._X(s["Sample Type"], s.v, s.age), s["Rt/R0"].values
        reg = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0)
        pred = cross_val_predict(reg, X, y, cv=KFold(5, shuffle=True, random_state=0))
        self.metrics = dict(n=len(y), r2=float(1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)),
                            mae=float(np.mean(np.abs(pred - y))))
        self.reg = reg.fit(X, y)

    def _X(self, mat, v, age):
        mat = np.asarray(mat)
        return np.column_stack([mat == m for m in self.materials] + [np.asarray(v, float), np.log10(np.asarray(age, float) + 1)]).astype(float)

    def predict(self, material, v, age):
        return float(self.reg.predict(self._X([material], [v], [age]))[0])


# ---------------------------------------------------------------- pooled FLEX + CFI-G
class PooledModel:
    """One self-extinction model across two experiments and two fuel types (liquid droplets, gas flames)."""
    def __init__(self):
        f = flex_data().dropna(subset=["o2", "p_atm"])
        g = cfig_flames()
        self.df = pd.concat([pd.DataFrame({"o2": f.o2, "p_atm": f.p_atm, "liquid": 1, "y": f.self_extinguished, "src": "FLEX"}),
                             pd.DataFrame({"o2": g.o2, "p_atm": g.p_atm, "liquid": 0, "y": g.self_ext, "src": "CFI-G"})],
                            ignore_index=True)
        X, y = self._X(self.df.o2, self.df.p_atm, self.df.liquid), self.df.y.values
        clf = o2_monotone_clf(3, 0)
        prob = cross_val_predict(clf, X, y, cv=KFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        self.metrics = dict(n=len(y), accuracy=float(np.mean((prob > 0.5) == y)), base_rate=float(max(y.mean(), 1 - y.mean())))
        self.clf = clf.fit(X, y)

    def _X(self, o2, p, liquid):
        return np.column_stack([np.asarray(o2, float), np.log10(np.asarray(p, float)), np.asarray(liquid, float)])

    def curve(self, o2s, p_atm, liquid):
        return self.clf.predict_proba(self._X(o2s, [p_atm] * len(o2s), [liquid] * len(o2s)))[:, 1]


REJECTED = [  # models tried and dropped because they did not beat simple guessing
    ("Gas-flame ignition (CFI-G)", "no better than always guessing 'ignites'"),
    ("Gas-flame burn time (CFI-G)", "R² 0.31: too weak to show"),
    ("Solid burn intensity, CO₂ produced (BASS-II)", "R² 0.19: depends on test length, not conditions"),
]
