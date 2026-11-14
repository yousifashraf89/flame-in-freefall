"""
Flame in Freefall - PSI combustion data harvester
Run on a laptop with internet:  pip install requests pandas  &&  python harvest_psi.py
Output:
  data/investigations.json   - full metadata for every Combustion Science investigation
  data/investigations.csv    - one-row-per-investigation summary (for the dashboard + the LLM)
  data/tables/PSI-xx__<name>.csv - each investigation's "Experimental table" (the test matrix)
Only metadata + experimental tables are downloaded (a few MB). The raw video/image
data is multiple TB - do NOT download it.
Cite each investigation by its DOI (PSI requires it).
"""
import json, os, re, time
from urllib.parse import quote
import requests
import pandas as pd

BASE = "https://psi.nasa.gov/geode-py/ws"
OUT = "data"
os.makedirs(f"{OUT}/tables", exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "SpaceApps2026-FlameInFreefall/1.0"


def get(url, **kw):
    for attempt in range(3):
        try:
            r = S.get(url, timeout=60, **kw)
            r.raise_for_status()
            return r
        except Exception as e:
            print("  retry", attempt + 1, url, e)
            time.sleep(2)
    raise RuntimeError(url)


# 1. Find investigations (search 'combustion', keep Combustion Science only)
ids = set()
for term in ["combustion", "flame", "fire", "smoke"]:
    r = get(f"{BASE}/repo/search", params={"term": term, "source": "psi", "type": "investigation"})
    ids |= set(re.findall(r"PSI-\d+", r.text))
print(f"{len(ids)} candidate investigations")

records, summary = [], []
for pid in sorted(ids, key=lambda s: int(s.split("-")[1])):
    inv = get(f"{BASE}/repo/investigations/{pid}").json()
    if inv.get("researchArea") != "Combustion Science":
        continue
    print(pid, inv.get("investigationAcronym"), "-", inv.get("title"))
    records.append(inv)

    tables = []
    for t in inv.get("experimentalTable") or []:
        fname = t["file_name"]
        # API returns a short-lived presigned S3 URL as plain text
        signed = get(f"{BASE}/studies/{pid}/download", params={"file": fname, "redirect": "false"}).text.strip()
        raw = get(signed).content
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", fname.replace(f"{pid}_Experimental table_", ""))
        path = f"{OUT}/tables/{pid}__{safe}"
        with open(path, "wb") as f:
            f.write(raw)
        tables.append(path)

    summary.append({
        "id": pid,
        "acronym": inv.get("investigationAcronym"),
        "title": inv.get("title"),
        "project_type": inv.get("projectType"),          # Microgravity vs Ground
        "sub_area": inv.get("subResearchArea"),
        "flight_program": inv.get("flightProgram"),
        "hardware": inv.get("experimentHardware"),
        "start": inv.get("investigationStartDate"),
        "end": inv.get("investigationEndDate"),
        "doi": inv.get("doi"),
        "n_publications": len(inv.get("publications") or []),
        "data_size_GB": round((inv.get("totalFileSize") or 0) / 1e9, 1),
        "objective": inv.get("objective"),
        "hypothesis": inv.get("hypothesis"),
        "approach": inv.get("approach"),
        "research_impacts": inv.get("researchImpacts"),
        "publication_titles": " | ".join(p.get("title", "") for p in inv.get("publications") or []),
        "tables": ";".join(tables),
    })

json.dump(records, open(f"{OUT}/investigations.json", "w"), indent=1)
pd.DataFrame(summary).to_csv(f"{OUT}/investigations.csv", index=False)
print(f"\nDone: {len(summary)} combustion investigations -> {OUT}/")
