"""
Flame in Freefall - AI-assisted fire-safety insights from NASA PSI microgravity combustion data.
Run:  pip install -r requirements.txt  &&  streamlit run app.py
Data: data/investigations.csv, data/tests_normalized.csv, (optional) data/investigations_enriched.json
AI chat: set ANTHROPIC_API_KEY as an env var or in .streamlit/secrets.toml
"""
import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from coverage_map import build_figure
from scoring import CURATED, HABITATS, WEIGHTS, score

DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="Flame in Freefall", page_icon="🔥", layout="wide")


@st.cache_data
def load():
    inv = pd.read_csv(DATA / "investigations.csv")
    tests = pd.read_csv(DATA / "tests_normalized.csv")
    p = DATA / "investigations_enriched.json"
    enriched = json.load(open(p)) if p.exists() else {}
    return inv, tests, enriched


inv, tests, enriched = load()
psi_url = lambda pid: f"https://psi.nasa.gov/physci/repo/data/investigations/{pid}"
doi_url = lambda doi: f"https://doi.org/{doi}" if isinstance(doi, str) and doi else ""
name_of = {r.id: (r.acronym if isinstance(r.acronym, str) else r.title[:45]) for r in inv.itertuples()}

st.title("🔥 Flame in Freefall")
st.caption("Fire-safety insights for crewed exploration from NASA's Physical Sciences Informatics (PSI) "
           f"combustion data · {len(inv)} investigations · {len(tests):,} tests")

# ---------------- sidebar: habitat ----------------
with st.sidebar:
    st.header("Crew environment")
    hab = st.selectbox("Habitat", list(HABITATS) + ["Custom…"])
    o2_def, p_def = HABITATS.get(hab, (0.30, 0.69))
    o2 = st.slider("O2 (%)", 10, 100, int(round(o2_def * 100))) / 100
    p_psi = st.slider("Pressure (psi)", 3.0, 20.0, round(p_def * 14.696, 1), 0.1)
    p_atm = p_psi / 14.696
    with st.expander("Score weights"):
        w = {k: st.slider(k, 0.0, 1.0, v, 0.05) for k, v in WEIGHTS.items()}
        tot = sum(w.values()) or 1
        w = {k: v / tot for k, v in w.items()}
    st.caption("Moon-base values (≥30% O2, ≤10.2 psi) are from the challenge statement. "
               "Mars is assumed equal to the Moon base.")

ranked = score(inv, tests, o2, p_atm, w)
tab_rank, tab_map, tab_cmp, tab_exp, tab_ask, tab_gap = st.tabs(
    ["🏆 Ranked for this habitat", "🗺️ Coverage map", "⚖️ Compare", "🔬 Data explorer", "💬 Ask the data", "🕳️ Gaps & Earth"])

# ---------------- 1. ranking ----------------
with tab_rank:
    st.subheader(f"Most relevant investigations for {o2:.0%} O2 at {p_psi:.1f} psi")
    top = ranked.head(15)
    fig = go.Figure()
    for k, label, color in [("pillar", "Fire-safety focus", "#d1495b"), ("envelope", "Closeness of test conditions", "#edae49"),
                            ("fuel", "Fuel relevance", "#00798c"), ("data", "Data richness", "#30638e")]:
        fig.add_bar(y=top.name, x=top[f"pts_{k}"], name=label, orientation="h", marker_color=color)
    fig.update_layout(barmode="stack", height=480, yaxis=dict(autorange="reversed"), xaxis_title="Relevance score (0-100)",
                      template="plotly_white", legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    for r in top.itertuples():
        meta = inv[inv.id == r.id].iloc[0]
        e = enriched.get(r.id, {})
        with st.expander(f"**{r.score:.0f}** · {r.name} ({r.id}) — {r.pillars} · {r.fuel_phase}"):
            st.write(e.get("one_line_summary") or str(meta.objective)[:400] + "…")
            if e.get("habitat_relevance"):
                st.info("AI summary: " + e["habitat_relevance"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Tests in PSI table", r.n_tests)
            c2.metric("Publications", int(meta.n_publications))
            c3.metric("Nearest test closeness", f"{r.E:.2f}" if r.E_known else "unknown")
            if r.linked_to:
                st.caption(f"Ground study reusing flight data from {name_of.get(r.linked_to, r.linked_to)} ({r.linked_to}).")
            st.markdown(f"[PSI page]({psi_url(r.id)})" + (f" · DOI [{meta.doi}]({doi_url(meta.doi)})" if doi_url(meta.doi) else ""))

# ---------------- 2. coverage map ----------------
with tab_map:
    d = tests.dropna(subset=["o2_fraction", "pressure_atm"])
    near = d[(d.o2_fraction.sub(o2).abs() <= 0.03) & ((d.pressure_atm / p_atm).between(0.8, 1.25))]
    moon = d[(d.o2_fraction >= 0.30) & (d.pressure_atm * 14.696 <= 10.2)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Tests with known O2 + pressure", f"{len(d):,}")
    c2.metric("Inside Moon-base envelope", len(moon), f"{moon.fuel_phase.str.startswith('solid').sum()} with solid materials",
              delta_color="off")
    c3.metric(f"Within ±3% O2 / ±20% p of your selection", len(near))
    fig = build_figure(tests)
    fig.add_trace(go.Scatter(x=[o2 * 100], y=[p_psi], mode="markers", name="Your selection",
                             marker=dict(symbol="x", size=16, color="#e63946", line=dict(width=2))))
    st.plotly_chart(fig, width="stretch")
    st.caption("Hollow markers: O2/pressure not recorded in the PSI table; ISS cabin air assumed (≈21%, 1 atm). "
               "No solid-material test in PSI exceeds ~22% O2 — the conditions that matter most for a Moon-base cabin fire are untested.")
    if len(near):
        st.markdown("**Tests closest to your selection**")
        st.dataframe(near[["investigation", "psi_id", "fuel", "fuel_phase", "o2_fraction", "pressure_atm", "outcome", "burn_time_s"]],
                     width="stretch", hide_index=True)

# ---------------- 3. compare ----------------
with tab_cmp:
    pick = st.multiselect("Pick 2–3 investigations", ranked.id.tolist(), default=ranked.id.head(3).tolist(),
                          format_func=lambda i: f"{name_of[i]} ({i})", max_selections=3)
    if pick:
        def envelope(pid):
            t = tests[tests.psi_id == pid]
            if t.empty:
                return dict(o2_range="—", p_range="—", fuels="—")
            f = lambda s, k=1: f"{s.min() * k:.2g}–{s.max() * k:.2g}" if s.notna().any() else "not recorded"
            return dict(o2_range=f(t.o2_fraction, 100) + " %", p_range=f(t.pressure_atm, 14.696) + " psi",
                        fuels=", ".join(str(x) for x in t.fuel.value_counts().index[:5]))
        rows = {}
        for pid in pick:
            m, e, env = inv[inv.id == pid].iloc[0], enriched.get(pid, {}), envelope(pid)
            pillars, phase, _ = CURATED.get(pid, (["fundamental"], "gas", None))
            rows[f"{name_of[pid]} ({pid})"] = {
                "Fire-safety pillar": ", ".join(e.get("fire_safety_pillars", pillars)),
                "Type / hardware": f"{m.project_type} · {m.hardware}",
                "Fuel": e.get("fuel", phase) if e else ", ".join([phase, env["fuels"]]),
                "Sample geometry": e.get("sample_geometry", "run AI extraction"),
                "O2 tested": env["o2_range"], "Pressure tested": env["p_range"],
                "What varied": e.get("what_varied", "run AI extraction"),
                "Measurements": ", ".join(e.get("measurements", [])) or "run AI extraction",
                "Key findings": " • ".join(e.get("key_findings", [])) or str(m.research_impacts)[:300],
                "Tests / publications": f"{(tests.psi_id == pid).sum()} / {m.n_publications}",
                "Cite": m.doi,
            }
        st.dataframe(pd.DataFrame(rows), width="stretch", height=520)
        if not enriched:
            st.warning("Run `python extract_attributes.py` to fill the AI-extracted rows.")

# ---------------- 4. explorer ----------------
with tab_exp:
    sel = st.multiselect("Investigations", sorted(tests.investigation.unique()), default=["FLEX"])
    t = tests[tests.investigation.isin(sel)] if sel else tests
    num_cols = ["o2_fraction", "pressure_atm", "flow_cm_s", "burn_time_s"]
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("x", num_cols, 0)
    y = c2.selectbox("y", num_cols, 3)
    color = c3.selectbox("colour by", ["fuel", "diluent", "outcome", "investigation", "fuel_phase"])
    tt = t.dropna(subset=[x, y])
    fig = go.Figure()
    for k, g in tt.groupby(color):
        fig.add_scatter(x=g[x], y=g[y], mode="markers", name=str(k), text=g.test_id)
    fig.update_layout(template="plotly_white", xaxis_title=x, yaxis_title=y, height=450)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(t, width="stretch", hide_index=True)
    st.download_button("Download filtered CSV", t.to_csv(index=False), "psi_tests_filtered.csv")

# ---------------- 5. ask (retrieval + LLM with citations) ----------------
def docs():
    out = []
    for r in inv.fillna("").itertuples():
        t = tests[tests.psi_id == r.id]
        env = (f"Tests: {len(t)}; O2 {t.o2_fraction.min():.2f}-{t.o2_fraction.max():.2f}; "
               f"pressure {t.pressure_atm.min():.2f}-{t.pressure_atm.max():.2f} atm; fuels: {', '.join(str(x) for x in t.fuel.dropna().unique()[:8])}"
               if len(t) and t.o2_fraction.notna().any() else "No O2/pressure test table.")
        e = json.dumps(enriched.get(r.id, {}))
        out.append(f"[{r.id} | {name_of[r.id]} | DOI {r.doi}]\n{r.title}\nObjective: {r.objective}\n"
                   f"Impacts: {r.research_impacts}\n{env}\nAI attributes: {e}")
    return out


def retrieve(q, k=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    D = docs()
    v = TfidfVectorizer(stop_words="english").fit(D + [q])
    sims = (v.transform(D) @ v.transform([q]).T).toarray().ravel()
    return [D[i] for i in sims.argsort()[::-1][:k]]


with tab_ask:
    st.caption("Answers use only PSI data and cite PSI IDs / DOIs. If the data doesn't cover it, it says so.")
    q = st.chat_input("e.g. Which tests are closest to 34% O2 at 8 psi? What suppressed droplet flames best?")
    if q:
        st.chat_message("user").write(q)
        ctx = retrieve(q)
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            try:
                key = st.secrets.get("ANTHROPIC_API_KEY")
            except Exception:
                key = None
        with st.chat_message("assistant"):
            if not key:
                st.warning("No ANTHROPIC_API_KEY set — showing the most relevant investigations instead.")
                for c in ctx:
                    st.text(c[:700])
            else:
                import anthropic
                msg = anthropic.Anthropic(api_key=key).messages.create(
                    model=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"), max_tokens=900,
                    system=("You are a spacecraft fire-safety research assistant. Answer ONLY from the context. "
                            "Cite every claim as [PSI-xx]. If the context does not cover the question, say so plainly "
                            f"and name the gap. User habitat: {o2:.0%} O2 at {p_psi:.1f} psi."),
                    messages=[{"role": "user", "content": "CONTEXT:\n\n" + "\n\n".join(ctx) + f"\n\nQUESTION: {q}"}])
                st.markdown(msg.content[0].text)
            with st.expander("Sources used"):
                for c in ctx:
                    pid = c[1:].split(" |")[0]
                    st.markdown(f"- [{pid} · {name_of.get(pid, '')}]({psi_url(pid)})")

# ---------------- 6. gaps & Earth ----------------
with tab_gap:
    st.subheader("What the data can't tell you yet")
    st.markdown(f"""
- **No partial-gravity data in PSI.** Moon (1/6 g) and Mars (3/8 g) fires sit between the microgravity and 1-g results.
  Future: *Lunar-g Combustion Investigation (LUCI)*, *Flammability of Materials on the Moon (FM2)*.
- **Partial-gravity stand-in:** ACME E-FIELD Flames used an electric field acting on flame ions to mimic buoyancy.
- **µg vs 1-g bounds:** Saffire-II recorded burn/spread length in both microgravity and 1 g for the same materials.
- **Solids at high O2 are untested in µg:** {moon.fuel_phase.str.startswith('solid').sum()} solid-material tests in the Moon-base envelope.
""")
    st.subheader("Back on Earth")
    st.markdown("""
- Fine-water-mist extinguishers developed for the ISS · smoke detectors that tell smoke from dust (shuttle, 1970s; used in mining)
  · firefighter air tanks and protective clothing from spacesuit technology.
- Sources: [Fire detector](https://spinoff.nasa.gov/node/9407) · [Firefighter air tanks](https://spinoff.nasa.gov/node/9582) ·
  [Fire attire](https://spinoff.nasa.gov/node/9918)
""")
st.caption("Data: NASA PSI (CC0). Cite each investigation by its DOI. AI-generated text is labelled and may contain errors.")
