"""
Flame in Freefall - step 2: normalize every PSI test table into ONE schema.

Usage:  python normalize_psi.py psi_combustion_data
Output: psi_combustion_data/tests_normalized.csv

One row per test (or per listed condition for summary tables like FLEX-2 / BRE).
Columns:
  psi_id, investigation, test_id, fuel, fuel_phase (solid/liquid/gas),
  o2_fraction, pressure_atm, flow_cm_s, diluent, outcome, burn_time_s,
  o2_source, p_source   -> 'measured' | 'listed' (design list/range) | 'assumed_cabin_air'
"""
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "psi_combustion_data")
T = ROOT / "tables"
ATM_PER_BAR, MMHG_PER_ATM = 1 / 1.01325, 760.0


def read(name):
    df = pd.read_csv(T / name, encoding="latin-1")
    df.columns = [c.replace("ï»¿", "").strip() for c in df.columns]   # BOM + stray spaces
    return df


def num(x):
    """First number in a messy cell: '~ 21.5' -> 21.5, '21.5 - 21.7' -> 21.6 (midpoint), '1 atm' -> 1."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    vals = [float(v) for v in re.findall(r"\d*\.?\d+", str(x).replace(",", ""))]
    if not vals:
        return np.nan
    if re.search(r"\d\s*-\s*\d", str(x)) and len(vals) >= 2:
        return (vals[0] + vals[1]) / 2
    return vals[0]


def nums(x):
    """All numbers in a list cell: '0.17; 0.21;0.25' -> [0.17, 0.21, 0.25]."""
    return [float(v) for v in re.findall(r"\d*\.?\d+", str(x))] if pd.notna(x) else []


def frame(psi_id, inv, phase, **cols):
    df = pd.DataFrame(cols)
    df.insert(0, "fuel_phase", phase)
    df.insert(0, "investigation", inv)
    df.insert(0, "psi_id", psi_id)
    return df


rows = []

# ---------- liquid droplets (MDCA in CIR) ----------
d = read("PSI-69__FLEX.csv")
dil = np.select([d.iloc[:, 9] > 0, d.iloc[:, 8] > 0], ["He", "CO2"], "N2")
rows.append(frame("PSI-69", "FLEX", "liquid",
    test_id=d["FLEX Identifier"], fuel=d["Fuel"],
    o2_fraction=d.iloc[:, 6].map(num), pressure_atm=d["Ambient pressure; mmHg"].map(num) / MMHG_PER_ATM,
    flow_cm_s=0.0, diluent=dil, outcome=d["Test end"].str.strip(),
    burn_time_s=d["Burn time; s"].map(num), o2_source="measured", p_source="measured"))

d = read("PSI-39__CFI.csv")
rows.append(frame("PSI-39", "CFI", "liquid",
    test_id=d["CIR"], fuel=d["Fuel"], o2_fraction=d["Oxygen"].map(num),
    pressure_atm=d["Pressure"].map(num), flow_cm_s=0.0,
    diluent=np.where(d["Helium"].map(num) > 0, "He", "N2"), outcome=None, burn_time_s=np.nan,
    o2_source="measured", p_source="measured"))

d = read("PSI-68__Flex2.csv")   # design matrix: every listed O2 x pressure combination
recs = []
for _, r in d.iterrows():
    for o2 in nums(r["Oxygen Mole Fraction"]):
        for p in nums(r["Ambient Pressure (atm)"]):
            recs.append(dict(test_id=None, fuel=r["Sample (Fuel Type)"], o2_fraction=o2, pressure_atm=p,
                             flow_cm_s=num(r["Flow Velocity (cm/s)"]), diluent="N2 or He"))
rows.append(frame("PSI-68", "FLEX-2", "liquid", **pd.DataFrame(recs).to_dict("list"),
                  outcome=None, burn_time_s=np.nan, o2_source="listed", p_source="listed"))

# ---------- gaseous flames (ACME in CIR) ----------
for f, flame, o2col, fuelcol in [("PSI-10__Flame_Design_Normal_Flame.csv", "normal", "XO2 (init)", "Fuel"),
                                 ("PSI-10__Flame_Design_Inverse_Flame.csv", "inverse", "XO2", None)]:
    d = read(f)
    rows.append(frame("PSI-10", "ACME Flame Design", "gas",
        test_id=d["Test"], fuel=d[fuelcol] if fuelcol else "C2H4",
        o2_fraction=pd.to_numeric(d[o2col], errors="coerce"),
        pressure_atm=pd.to_numeric(d["pmean (atm)"], errors="coerce"), flow_cm_s=0.0,
        diluent="N2", outcome=flame + " flame", burn_time_s=np.nan, o2_source="measured", p_source="measured"))

for f, o2col in [("PSI-159__CFI-G_Normal_Flames.csv", "XO2 before test"), ("PSI-159__CFI-G_Inverse_Flames.csv", "XO2")]:
    d = read(f)
    term = [c for c in d.columns if c.lower().startswith("self")][0]
    out = d[term].astype(str).str.strip().map({"SE": "self-extinguished", "FT": "flow terminated"})
    out = out.where(d["Hot ignition"].astype(str).str.strip() != "No", "no ignition")
    rows.append(frame("PSI-159", "ACME CFI-G", "gas",
        test_id=d["Test"], fuel=d["Fuel"], o2_fraction=pd.to_numeric(d[o2col], errors="coerce"),
        pressure_atm=pd.to_numeric(d["approx p (bar)"], errors="coerce") * ATM_PER_BAR, flow_cm_s=0.0,
        diluent="N2", outcome=out, burn_time_s=pd.to_numeric(d["burn time (sec)"], errors="coerce"),
        o2_source="measured", p_source="measured"))

d = read("PSI-20__ACME_BRE.csv")   # O2 levels listed per test day; pressure varied but not tabulated
recs = [dict(test_id=f"{r['Year']}-{r['Month']}-{r['Date']}", fuel=r["Fuel Bottle"], o2_fraction=o2 / 100)
        for _, r in d.iterrows() for o2 in nums(r["Oxygen Conc (percent)"]) if o2 <= 100]
rows.append(frame("PSI-20", "ACME BRE", "gas", **pd.DataFrame(recs).to_dict("list"), pressure_atm=np.nan,
                  flow_cm_s=0.0, diluent="N2", outcome=None, burn_time_s=np.nan, o2_source="listed", p_source="unknown"))

d = read("PSI-23__ACME_s-Flame.csv")   # notes: atmosphere ~1 atm
rows.append(frame("PSI-23", "ACME s-Flame", "gas",
    test_id=d.iloc[:, 1].astype(str), fuel=d["Fuel Bottle"], o2_fraction=d["Atmosphere*"].map(num) / 100,
    pressure_atm=1.0, flow_cm_s=0.0, diluent=np.where(d["Test Type"].str.contains("helium", case=False, na=False), "He", "N2"),
    outcome=None, burn_time_s=np.nan, o2_source="listed", p_source="listed"))

# ---------- solids & gas jets in ISS cabin air / Cygnus (O2 measured where given, else cabin air) ----------
d = read("PSI-25__BASS-II.csv")
o2 = d["Calibrated  initial O2 % by vol"].map(num) / 100
rows.append(frame("PSI-25", "BASS-II", "solid",
    test_id=d["Test #"], fuel=d["Fuel Sample Material"].str.strip(), o2_fraction=o2, pressure_atm=1.0,
    flow_cm_s=np.nan, diluent="N2", outcome=None, burn_time_s=np.nan,
    o2_source=np.where(o2.notna(), "measured", "unknown"), p_source="assumed_cabin_air"))

d = read("PSI-26__BASS.csv")
rows.append(frame("PSI-26", "BASS", "solid", test_id=d["Test #"], fuel=d["Fuel Sample Material"].str.strip(),
    o2_fraction=0.21, pressure_atm=1.0, flow_cm_s=np.nan, diluent="N2", outcome=None, burn_time_s=np.nan,
    o2_source="assumed_cabin_air", p_source="assumed_cabin_air"))

for f, inv, pid, o2col in [("PSI-98__SAFFIRE-1.csv", "Saffire-I", "PSI-98", "Percent O2"),
                           ("PSI-99__SAFFIRE-2.csv", "Saffire-II", "PSI-99", "Percent O2 (Note 1)"),
                           ("PSI-100__SAFFIRE-3.csv", "Saffire-III", "PSI-100", None)]:
    d = read(f)
    d = d[d["Material"].notna() & ~d["Material"].astype(str).str.contains("visu", case=False)]
    o2 = d[o2col].map(num) / 100 if o2col else pd.Series(0.21, index=d.index)
    rows.append(frame(pid, inv, "solid", test_id=d["Sample Number"], fuel=d["Material"],
        o2_fraction=o2, pressure_atm=1.0, flow_cm_s=d["Air Flow (cm/s)" if "Air Flow (cm/s)" in d else "Air Flow"].map(num),
        diluent="N2", outcome=d.get("Flow Direction", d.get("Direction")), burn_time_s=d["Burn Time (s)"].map(num) if "Burn Time (s)" in d else np.nan,
        o2_source="measured" if o2col else "assumed_cabin_air", p_source="assumed_cabin_air"))

d = read("PSI-107__Spice.csv")
rows.append(frame("PSI-107", "SPICE", "gas", test_id=d["GMT date"].astype(str) + " " + d["GMT ignition"].astype(str),
    fuel=d["Fuel"], o2_fraction=0.21, pressure_atm=1.0, flow_cm_s=d["Air Rate (cm/s)"].map(num), diluent="N2",
    outcome=None, burn_time_s=np.nan, o2_source="assumed_cabin_air", p_source="assumed_cabin_air"))

d = read("PSI-106__Slice.csv")
rows.append(frame("PSI-106", "SLICE", "gas", test_id=d["Test Date"], fuel=d["Fuel Type"], o2_fraction=0.21,
    pressure_atm=1.0, flow_cm_s=np.nan, diluent="N2", outcome=None, burn_time_s=np.nan,
    o2_source="assumed_cabin_air", p_source="assumed_cabin_air"))

for f, pid, inv, mat, flow in [("PSI-102__SAME.csv", "PSI-102", "SAME", "Sample Material", "Air Flow (cm/s)"),
                               ("PSI-101__SAME-R.csv", "PSI-101", "SAME-R", "Sample Type", "Inlet velocity (cm/s)")]:
    d = read(f)
    rows.append(frame(pid, inv, "solid (smoke)", test_id=d.iloc[:, 0 if pid == "PSI-102" else 1].astype(str),
        fuel=d[mat].astype(str).str.strip(), o2_fraction=0.21, pressure_atm=1.0, flow_cm_s=d[flow].map(num),
        diluent="N2", outcome="smoke sample", burn_time_s=np.nan,
        o2_source="assumed_cabin_air", p_source="assumed_cabin_air"))

# Not mapped (no O2/pressure in table): PSI-21 CLD Flame, PSI-22 E-FIELD, PSI-47 DAFT (dust), PSI-142 (figure index)

out = pd.concat(rows, ignore_index=True)
out["fuel"] = out["fuel"].astype(str).str.strip()
# sanity: physically impossible values -> blank
out.loc[~out["pressure_atm"].between(0.05, 10), "pressure_atm"] = np.nan
out.loc[~out["o2_fraction"].between(0.05, 1.0), "o2_fraction"] = np.nan
out.to_csv(ROOT / "tests_normalized.csv", index=False)
print(f"{len(out)} rows -> {ROOT / 'tests_normalized.csv'}")
print(out.groupby(["investigation"]).agg(n=("psi_id", "size"),
      o2_min=("o2_fraction", "min"), o2_max=("o2_fraction", "max"),
      p_min=("pressure_atm", "min"), p_max=("pressure_atm", "max")).round(3).to_string())
