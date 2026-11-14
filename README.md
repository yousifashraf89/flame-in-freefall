# 🔥 Flame in Freefall: NASA Space Apps 2026

**Challenge:** Flame in Freefall: SI-Powered Fire Safety Insights from Microgravity Combustion Data

A dashboard that turns all 24 combustion investigations in NASA's Physical Sciences Informatics (PSI) database into
fire-safety answers for crewed habitats (ISS, Moon base and rovers, spacesuits, Mars).

## How it works (guided flow)
1. **Choose a location:** ISS, Lunar Gateway, Moon base, pressurized rover, Mars transit ship, Mars base, or spacesuit.
2. **Choose a task:**
   - 🔥 **How risky is fire here?** Verdict, coverage map, most relevant experiments, protections, next tests, PDF report
   - 🏗️ **Check my module's materials.** Fire-risk card with a traffic light per material
   - 📈 **Predict a fire.** Four ML models from three NASA experiments (liquid, gas, smoke, combined), with validation
   - 🕳️ **Where is NASA's data missing?** Coverage maps for each fuel type, the next tests NASA should run, "will my material burn here?"
   - 🧯 **What protects the crew?** Suppression (BASS measured outcomes, BASS-II flow-to-extinction runs), detection (smoke), material choice (Saffire)
   - 📜 **What NASA has learned.** Past spacecraft fires (Apollo 1, Apollo 13, Mir), seven key ISS lessons, upcoming experiments (FM2, SoFIE, FLARE)
   - ⚖️ **Choose the best atmosphere.** Pre-breathe vs NFPA 99B increased burning vs data coverage
   - 📚 **Explore NASA's data.** Library of all 24 investigations, comparisons, plots, and a download of the full cleaned dataset
   - 💬 **Ask a question.** Answers grounded in PSI, the knowledge base and the NASA Researcher's Guide, with clickable citations. Without ANTHROPIC_API_KEY it shows the best-matching passages.
   - Footer: 🌍 **Back on Earth** (spinoffs) and ℹ️ **About** (glossary, methods, sources)

Every page starts with a short "What this page tells you" box, and nothing from v3 was removed.

**Key findings:** of 2,027 tests with recorded O₂ and pressure, only 74 fall in the planned Moon-base atmosphere, and **none of
them used solid materials**. In FLEX and CFI-G, flames become much less likely to go out on their own as oxygen rises.

## Models (5-fold cross-validation)
| Model | NASA data | Tests | Accuracy | Guessing the most common outcome |
|---|---|---|---|---|
| Liquid fuel: burning rate / lifetime | FLEX (PSI-69) | 257 / 258 | R² 0.65 / 0.59 | — |
| Liquid fuel: goes out on its own | FLEX (PSI-69) | 261 | 82% | 69% |
| Gas flame: goes out on its own | ACME CFI-G (PSI-159), normal flames | 132 | 89% | 77% |
| Smoke particle growth | SAME-R (PSI-101) | 109 | R² 0.89 | — |
| All fuels: goes out on its own | FLEX + CFI-G | 405 | 77% | 71% |

Self-extinction models are physics-constrained (monotonic in oxygen). Tried and dropped: CFI-G ignition (no skill),
CFI-G burn time (R² 0.31), BASS-II CO₂ produced (R² 0.19).

## Extra data collected from the challenge resources (v5)
| File | What it is | Source |
|---|---|---|
| `data/extra/bass_as_run.csv` | 106 BASS tests with measured O₂, flow and outcome (adds outcomes to the BASS rows) | PSI-26 (BASS) As-Run test matrix |
| `data/extra/bass2_flow_runs.csv` | 129 BASS-II runs, 31 with flow reduced until the flame went out | PSI-25 (BASS-II) test matrices |
| `data/extra/cool_flame_extinction.csv` | Hot- and cool-flame extinction sizes for droplets | PSI-117 analysed FLEX data (Proc. Combust. Inst. 2023) |
| `data/extra/flex_droplet_traces.csv` | Droplet size over time (d² law) for sample FLEX tests | PSI-69 (FLEX) raw droplet data |
| `data/extra/guide_chunks.json` | 86 passages from the NASA Researcher's Guide to ISS Combustion Science | nasa.gov (challenge resource) |
| `data/knowledge.json` | 13 curated lessons, future experiments and historic fires, each with a source link | Challenge resource pages + NASA reports |

**Correction we made:** NASA's FM2 work suggests some materials burn at *lower* oxygen in lunar gravity than on Earth,
so partial gravity can be the worst case. The app says so; it does not assume Moon fire risk lies "between" ISS and Earth.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py
Raw tables can sit in data/raw/ or anywhere in the repo; the app finds them. `.streamlit/config.toml` is optional.

## Data pipeline (already run; outputs are in data/)
1. `harvest_psi.py` downloads PSI metadata + experimental tables (public PSI API)
2. `normalize_psi.py psi_combustion_data` builds `data/tests_normalized.csv` (one schema for 2,961 tests; BASS rows enriched from data/extra/bass_as_run.csv)
3. `extract_attributes.py` (optional; needs ANTHROPIC_API_KEY) builds `data/investigations_enriched.json`
   with AI-extracted attributes, which the app shows labelled "(AI)". Have a person review it.
Raw tables used by the model and the equipment charts are in `data/raw/`.

## Methods & assumptions
- **Relevance score** = 35% fire-safety focus + 30% closeness of tested conditions + 20% fuel relevance + 15% data richness.
- **Coverage** = exp(−d/0.15), where d is the distance to the nearest test in (O₂ fraction, log₁₀ pressure).
- **Model:** gradient-boosted trees (scikit-learn) with quantile regression for 10–90% bands. 5-fold cross-validation:
  burning rate R² ≈ 0.65; self-extinction 82% accuracy vs 69% baseline. A CFI-G ignition model was tested and
  dropped (no skill over the baseline).
- Where PSI tables omit O₂ or pressure, ISS cabin air (≈21%, 14.7 psi) is assumed and flagged (`o2_source`, `p_source`).

## Use of AI (Super Intelligence)
- Parts of the code were written with the help of Claude (Anthropic) and reviewed by the team.
- AI summaries (`investigations_enriched.json`) and chat answers are generated by Claude and labelled in the app.
- Habitat briefings and PDF reports are rule-based (no AI model).
- No AI-generated images. The Space Apps logo and BASS photo are official files, shown unmodified.

## Sources beyond PSI
- Oxygen-enriched atmosphere definitions (NFPA 99, OSHA: >23.5% O₂): https://wha-international.com/what-is-an-oxygen-enriched-atmosphere/
- NFPA 99B "atmosphere of increased burning" (O₂% > 23.45/√P_atm): Beeson, NASA WSTF, ASTM STP: https://store.astm.org/stp12040s.html
- Hyperbaric practice and fires: https://wha-international.com/?p=1962 ·
  https://nbdhmt.org/wp-content/uploads/2026/03/Catastrophic-Hyperbaric-Chamber-Fire-Report-dated-March-27-2026.pdf ·
  https://baromedical.com/_files/ugd/e43c3a_e36f07ddab924991884056bf2a67bf24.pdf
- Branding: Space Apps logo, colours and fonts from https://www.spaceappschallenge.org/brand/ ; hero photo (NASA, BASS experiment)
  from the challenge page. Both are loaded unmodified from the Space Apps website.
- The pre-breathe index ((1−O₂)·P_cabin / 4.3 psi) is a simplified relative measure, not a medical criterion.

## Data & citation
NASA Physical Sciences Informatics, https://psi.nasa.gov (CC0). Cite each investigation by its DOI (listed in the app).
