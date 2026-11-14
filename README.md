# Flame in Freefall: NASA Space Apps 2026

An AI-assisted dashboard that ranks NASA PSI microgravity combustion investigations by relevance to crew habitats
(ISS, Moon base, rovers, spacesuit, Mars) and shows where the test data does and doesn't cover them.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Data pipeline (already run; outputs are in data/)
1. `python harvest_psi.py` downloads PSI metadata + experimental tables (needs internet access to psi.nasa.gov)
2. `python normalize_psi.py psi_combustion_data` builds data/tests_normalized.csv (one schema for all tests)
3. `python extract_attributes.py` (needs ANTHROPIC_API_KEY) builds data/investigations_enriched.json.
   Have a person check it before the demo.

## AI chat
Set `ANTHROPIC_API_KEY` as an environment variable, or in `.streamlit/secrets.toml`:
    ANTHROPIC_API_KEY = "sk-ant-..."
Without a key, the chat tab shows the retrieved investigations instead.

## Deploy (free)
Push this folder to GitHub, then go to share.streamlit.io, choose New app, pick app.py and add the key under Secrets.

## Data & citation
NASA Physical Sciences Informatics, https://psi.nasa.gov (CC0). Cite each investigation by its DOI.
Assumed values are marked in tests_normalized.csv (o2_source / p_source = assumed_cabin_air).
