"""
Predictive models trained on PSI microgravity data.
  1. FLEX droplet fires (PSI-69): burning rate (median + 10-90% band), burn time, P(self-extinction)
  (A CFI-G ignition model was tried and dropped: it did not beat the base rate.)
Every prediction carries an extrapolation check: how far the query is from the nearest real test.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
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
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0)
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
