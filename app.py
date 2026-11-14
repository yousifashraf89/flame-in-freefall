"""
Flame in Freefall: fire-safety insights for crewed exploration from NASA PSI microgravity combustion data.
Run:  pip install -r requirements.txt  &&  streamlit run app.py
AI chat: set ANTHROPIC_API_KEY (env var or .streamlit/secrets.toml)
Built with AI assistance (Claude, Anthropic); see the About page.
"""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from coverage_map import PHASE_COLORS, build_figure
from features import (MATERIALS, PHASE_GROUPS, SAFFIRE2, briefing, coverage, coverage_grid, gap_table, material_tests,
                      material_verdict, pdf_report, suppression_summary)
from models import FlexModel, reliability
from designer import EARTH, INVENTORY, assess, atmosphere_flags, earth_figure, guidance, prebreathe_index, tradeoff_figure
from scoring import CURATED, HABITATS, WEIGHTS, score

DATA = Path(__file__).parent / "data"
PSI = 14.696
st.set_page_config(page_title="Flame in Freefall", page_icon="🔥", layout="wide")

# ------------------------------------------------------------------ style
LOGO = "https://assets.spaceappschallenge.org/media/images/Space_Apps_Logo_Color_and_White.width-440.jpegquality-60.png"
BASS_PHOTO = "https://assets.spaceappschallenge.org/media/images/Flame_in_Freefall2_4.3.2e16d0ba.fill-785x660.jpg"
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;600;800;900&family=Overpass:wght@400;600;700&display=swap');
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th, .stCaption, [data-testid="stWidgetLabel"] p
  {font-family: 'Overpass', 'Fira Sans', sans-serif;}
h1, h2, h3, h4, .kpi .v, .hero h1 {font-family: 'Fira Sans', sans-serif !important; font-weight: 800 !important;}
.block-container {padding-top: 3.4rem; max-width: 1280px;}
.hero {background: linear-gradient(135deg, #0050d8 0%, #1b2b85 55%, #151622 100%); border-radius: 18px;
       padding: 26px 30px; margin-bottom: 18px; display:flex; gap:26px; align-items:center; border:1px solid #2b3aa0;}
.hero .txt {flex: 1.6;}
.hero h1 {font-size: 2.6rem; margin: 0 0 .35rem 0; color:#fff;}
.hero .tag {color:#e7f434; font-weight:700; letter-spacing:.04em; text-transform:uppercase; font-size:.85rem;}
.hero p {font-size: 1.08rem; color: #dfe1e2; margin: .4rem 0 0 0;}
.hero .img {flex: 1; text-align:right;}
.hero .img img {max-width: 100%; max-height: 230px; border-radius: 14px; box-shadow: 0 8px 30px rgba(0,0,0,.45);}
.hero .credit {font-size:.72rem; color:#a9b3d6; margin-top:4px;}
.kpi {background:#1c1f33; border:1px solid #2b3170; border-radius:14px; padding:16px 18px; height:100%;}
.kpi .v {font-size: 2.1rem; line-height: 1.1;}
.kpi .l {color:#b8bfd8; font-size:.9rem; margin-top:4px;}
.kpi.red .v {color:#ff5d5d;} .kpi.orange .v {color:#ff8a3d;} .kpi.blue .v {color:#2491ff;} .kpi.green .v {color:#3DDC84;}
.kpi.yellow .v {color:#e7f434;}
.card {background:#1c1f33; border:1px solid #2b3170; border-radius:14px; padding:14px 16px; margin-bottom:10px;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-weight:700; font-size:.85rem; margin-right:6px;}
.b-score {background:#e7f434; color:#151622;}
.b-tag {background:#2b3170; color:#dfe1e2; font-weight:500;}
.small {color:#a9b3d6; font-size:.88rem;}
.brief {background:linear-gradient(90deg, rgba(0,80,216,.25), rgba(27,43,133,.15)); border-left:4px solid #e7f434;
        border-radius:10px; padding:14px 18px; margin: 6px 0 16px 0;}
.risk {border-radius:12px; padding:10px 14px; margin-bottom:8px; border:1px solid #2b3170; background:#1c1f33;}
.risk.green {border-left:6px solid #3DDC84;} .risk.yellow {border-left:6px solid #e7f434;}
.risk.orange {border-left:6px solid #ff8a3d;} .risk.red {border-left:6px solid #ff5d5d;}
a {color:#2491ff !important;}
</style>""", unsafe_allow_html=True)

PILLAR_ICON = {"prevention": "🛡️ Prevention", "detection": "🔔 Detection", "suppression": "🧯 Suppression",
               "fundamental": "🔬 Fundamental"}
PHASE_ICON = {"solid": "🧱 Solid", "liquid": "💧 Liquid", "gas": "💨 Gas", "gas (solid emulator)": "💨 Gas (mimics a solid)",
              "dust": "🌫️ Dust"}


def kpi(col, value, label, tone="blue"):
    col.markdown(f'<div class="kpi {tone}"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def dark(fig, h=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                      height=h, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    inv = pd.read_csv(DATA / "investigations.csv")
    tests = pd.read_csv(DATA / "tests_normalized.csv")
    p = DATA / "investigations_enriched.json"
    return inv, tests, (json.load(open(p)) if p.exists() else {})


@st.cache_resource
def flex_model():
    return FlexModel()


inv, tests, enriched = load()
fm = flex_model()
from scoring import SHORT_NAMES
name_of = {r.id: SHORT_NAMES.get(r.id) or (r.acronym if isinstance(r.acronym, str) else r.title[:45]) for r in inv.itertuples()}
psi_url = lambda pid: f"https://psi.nasa.gov/physci/repo/data/investigations/{pid}"
known = tests.dropna(subset=["o2_fraction", "pressure_atm"])
moon = known[(known.o2_fraction >= 0.30) & (known.pressure_atm * PSI <= 10.2)]
moon_solids = int(moon.fuel_phase.str.startswith("solid").sum())

# ------------------------------------------------------------------ sidebar: crew environment
with st.sidebar:
    st.markdown(f'<img src="{LOGO}" style="width:170px;margin:-10px 0 4px 0"><div class="small">Built for the NASA Space Apps '
                'Challenge 2026</div>', unsafe_allow_html=True)
    st.markdown("### 🧑‍🚀 Crew environment")
    hab = st.selectbox("Where will the crew live?", list(HABITATS),
                       help="Values come from the challenge statement. Mars is assumed to match the Moon base.")
    o2_def, p_def = HABITATS[hab]
    o2 = st.slider("Oxygen in the air (%)", 15, 100, int(round(o2_def * 100)), key=f"o2_{hab}",
                   help="Earth air is 21%. More oxygen makes materials ignite and burn more easily.") / 100
    p_psi = st.slider("Cabin pressure (psi)", 3.0, 16.0, round(p_def * PSI, 1), 0.1, key=f"p_{hab}",
                      help="Sea level on Earth is 14.7 psi. Spacesuits run at 4.3 psi.")
    p_atm = p_psi / PSI
    with st.expander("⚖️ How the relevance score is weighted"):
        w = {k: st.slider(lbl, 0.0, 1.0, WEIGHTS[k], 0.05) for k, lbl in
             [("pillar", "Fire-safety focus"), ("envelope", "Closeness of test conditions"),
              ("fuel", "Fuel relevance"), ("data", "Data richness")]}
        tot = sum(w.values()) or 1
        w = {k: v / tot for k, v in w.items()}
    st.caption(f"Data: NASA Physical Sciences Informatics · {len(inv)} investigations · {len(tests):,} tests")

ranked = score(inv, tests, o2, p_atm, w)
flex_here = fm.predict("Methanol", min(o2, 0.40), p_atm)

# ------------------------------------------------------------------ navigation
PAGES = ["🏠 Overview", "🧑‍🚀 Your habitat", "🏗️ Designer", "🕳️ Data gap", "⚖️ Trade-off", "📈 Predict",
         "🧯 Equipment", "🌍 Earth", "🔬 Explore", "💬 Ask", "ℹ️ About"]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]
go_to = lambda p: st.session_state.update(page=p)
page = st.segmented_control("Navigate", PAGES, key="page", label_visibility="collapsed") or PAGES[0]

# ================================================================== OVERVIEW
if page == "🏠 Overview":
    st.markdown(f"""<div class="hero"><div class="txt"><div class="tag">NASA Space Apps 2026 · Flame in Freefall</div>
    <h1>🔥 Flame in Freefall</h1>
    <p>Fire behaves differently in space: with no gravity there's no rising hot air, so a candle flame becomes a
    blue sphere. NASA has run decades of microgravity fire experiments. This dashboard turns them into
    <b>fire-safety answers for the Moon, Mars and beyond</b>, and lets you <b>design a safer habitat</b>.</p></div>
    <div class="img"><img src="{BASS_PHOTO}" alt="Flame from a NASA BASS experiment">
    <div class="credit">A flame in microgravity · Image: NASA (BASS experiment)</div></div></div>""", unsafe_allow_html=True)
    c = st.columns(4)
    kpi(c[0], len(inv), "NASA combustion investigations, combined in one place", "blue")
    kpi(c[1], f"{len(tests):,}", "individual fire tests, cleaned into one table", "green")
    kpi(c[2], "≥30%", "oxygen planned for Moon-base air (Earth: 21%)", "orange")
    kpi(c[3], moon_solids, f"solid-material tests at Moon-base conditions (of {len(moon)} tests there)", "red")
    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Why it matters")
        st.markdown("""
- **Apollo 1 (1967)**: a cabin fire in pure oxygen killed three astronauts during a ground test.
- **Mir (1997)**: an oxygen generator fire filled the station with smoke.
- **Artemis**: Moon bases and rovers will run at **higher oxygen and lower pressure** than the ISS, so materials will burn more easily.

NASA's results sit in 24 separate datasets with different formats. **Flame in Freefall** ranks them for your habitat,
shows where the evidence runs out, and predicts how a fire would behave.""")
    with right:
        st.markdown("#### Start here")
        st.button("🧑‍🚀  Which experiments matter for my habitat?", on_click=go_to, args=("🧑‍🚀 Your habitat",), width="stretch")
        st.button("🏗️  Design a Moon-base module and check its fire risk", on_click=go_to, args=("🏗️ Designer",), width="stretch",
                  type="primary")
        st.button("🕳️  Where is NASA's data missing?", on_click=go_to, args=("🕳️ Data gap",), width="stretch")
        st.button("⚖️  The atmosphere trade-off: spacewalks vs fire risk", on_click=go_to, args=("⚖️ Trade-off",), width="stretch")
        st.button("📈  Predict how a fire would burn", on_click=go_to, args=("📈 Predict",), width="stretch")
        st.button("💬  Ask a question in plain English", on_click=go_to, args=("💬 Ask",), width="stretch")
        st.caption("Pick a habitat in the sidebar; every page updates.")
    st.plotly_chart(build_figure(tests), width="stretch")
    st.caption("Each dot is one microgravity fire test. Stars mark where crews will live. The orange box is the planned "
               "Moon-base atmosphere: almost empty, and with no solid materials tested.")

# ================================================================== YOUR HABITAT
elif page == "🧑‍🚀 Your habitat":
    st.markdown(f"## 🧑‍🚀 {hab.split('(')[0].strip()}: {o2:.0%} O₂ at {p_psi:.1f} psi")
    import re as _re
    _b = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", briefing(hab.split("(")[0].strip(), o2, p_psi, ranked, tests, flex_here))
    st.markdown(f'<div class="brief">{_b}'
                f'<div class="small" style="margin-top:6px">Generated automatically from the PSI data.</div></div>',
                unsafe_allow_html=True)
    top = ranked.head(12)
    fig = go.Figure()
    for k, label, color in [("pillar", "Fire-safety focus", "#FF5D5D"), ("envelope", "Closeness of test conditions", "#FF9A4D"),
                            ("fuel", "Fuel relevance", "#3DDC84"), ("data", "Data richness", "#5AA9FF")]:
        fig.add_bar(y=top.name, x=top[f"pts_{k}"], name=label, orientation="h", marker_color=color,
                    hovertemplate="%{y}: %{x:.0f} pts<extra>" + label + "</extra>")
    fig.update_layout(barmode="stack", yaxis=dict(autorange="reversed"), xaxis_title="Relevance score (0–100)",
                      legend=dict(orientation="h", y=-0.13, x=0), title="Most relevant investigations, and why")
    st.plotly_chart(dark(fig, 470), width="stretch")
    st.caption("Score = 35% fire-safety focus + 30% how close the tests came to this atmosphere + 20% fuel relevance "
               "(solids count most) + 15% amount of data. Change the weights in the sidebar.")

    cols = st.columns(3)
    for i, r in enumerate(top.head(9).itertuples()):
        meta = inv[inv.id == r.id].iloc[0]
        e = enriched.get(r.id, {})
        summary = e.get("one_line_summary") or str(meta.objective).split(". ")[0]
        if len(summary) > 150:
            summary = summary[:150].rsplit(" ", 1)[0] + "…"
        tags = " ".join(f'<span class="badge b-tag">{PILLAR_ICON.get(p, p)}</span>' for p in r.pillars.split(", "))
        link = f' · ground study using {name_of.get(r.linked_to, r.linked_to)} data' if r.linked_to else ""
        cols[i % 3].markdown(
            f'<div class="card"><span class="badge b-score">{r.score:.0f}</span><b>{r.name}</b> '
            f'<span class="small">({r.id})</span><br>{tags}<span class="badge b-tag">{PHASE_ICON.get(r.fuel_phase, r.fuel_phase)}</span>'
            f'<p style="margin:8px 0 6px 0">{summary}</p><div class="small">{r.n_tests} tests · {int(meta.n_publications)} papers{link} · '
            f'<a href="{psi_url(r.id)}" target="_blank">PSI page</a>'
            + (f' · <a href="https://doi.org/{meta.doi}" target="_blank">DOI</a>' if isinstance(meta.doi, str) and meta.doi else "")
            + "</div></div>", unsafe_allow_html=True)
    with st.expander("See all investigations"):
        st.dataframe(ranked[["score", "name", "id", "pillars", "fuel_phase", "n_tests"]], hide_index=True, width="stretch")
    gaps = gap_table(tests)
    st.download_button("📄 Download a one-page brief for this habitat (PDF)",
                       pdf_report(hab, o2, p_psi, ranked, gaps, briefing(hab, o2, p_psi, ranked, tests, flex_here), fm.metrics),
                       file_name="flame_in_freefall_brief.pdf", mime="application/pdf", type="primary")

# ================================================================== DESIGNER
elif page == "🏗️ Designer":
    st.markdown("## 🏗️ Moon-base fire-risk designer")
    st.markdown("Pick the atmosphere in the sidebar and choose what's on board. The designer checks every item against "
                "NASA's microgravity fire data and returns a **fire-risk card** for your module.")
    a, b = st.columns([1, 1.5])
    with a:
        module = st.text_input("Module name", "Artemis Base Camp habitat module")
        items = st.multiselect("Materials on board", list(INVENTORY), default=list(INVENTORY)[:6],
                               help="Each item is matched to the closest material actually tested in PSI.")
        crew = st.number_input("Crew", 1, 8, 4)
    df, flags, fx, ready = assess(tests, items, o2, p_psi, fm)
    with b:
        c = st.columns(3)
        kpi(c[0], f"{ready:.0%}", "of on-board materials have microgravity fire data at this atmosphere",
            "green" if ready > 0.7 else "orange" if ready > 0.3 else "red")
        kpi(c[1], f"{flags['pp_o2_psi']:.1f} psi", "oxygen partial pressure (Earth sea level: 3.1)", "blue")
        kpi(c[2], f"{flags['prebreathe_index']:.2f}", "pre-breathe index (ISS cabin 2.70; lower = shorter pre-breathe)", "yellow")
        if flags["increased_burning"]:
            st.error(f"🔥 **Atmosphere of increased burning** (NFPA 99B): {o2:.0%} O₂ is above the {flags['limit_pct']:.1f}% "
                     f"limit for {p_psi:.1f} psi. Materials ignite more easily and burn faster than on Earth.")
        elif flags["oea"]:
            st.warning("⚠️ **Oxygen-enriched atmosphere** (above 23.5% O₂, NFPA 99 / OSHA).")
        else:
            st.success("✅ Oxygen level is within normal-air limits.")
    st.markdown(f"### 🪪 Fire-risk card: {module} ({o2:.0%} O₂ · {p_psi:.1f} psi · crew {crew})")
    cols = st.columns(2)
    for i, r in enumerate(df.itertuples()):
        extra = f" · {r.n} PSI tests" + (f", up to {r.max_o2:.0f}% O₂" if r.max_o2 else "")
        cols[i % 2].markdown(f'<div class="risk {r.status}"><b>{r.item}</b><br>{r.label}<span class="small">{extra}</span>'
                             f'<br><span class="small">Closest tested material: {r.proxy}. {r.note}</span></div>',
                             unsafe_allow_html=True)
    g = guidance(df, flags, fx, o2)
    t1, t2, t3, t4 = st.columns(4)
    lvl, msg = reliability(fx["distance"])
    t1.markdown(f'<div class="card"><b>🔥 How a fire would behave</b><br><span class="small">FLEX model, methanol droplet:'
                f' burns at ≈{fx["burn_rate"]["mid"]:.2f} mm²/s, goes out on its own {fx["p_self_ext"]:.0%} of the time. '
                f'{msg}</span></div>', unsafe_allow_html=True)
    t2.markdown(f'<div class="card"><b>🔔 Detection</b><br><span class="small">{g["detection"]}</span></div>', unsafe_allow_html=True)
    t3.markdown(f'<div class="card"><b>🧯 Suppression</b><br><span class="small">{g["suppression"]}</span></div>', unsafe_allow_html=True)
    t4.markdown('<div class="card"><b>🧪 Test before flight</b><br><span class="small">' +
                ("<br>".join("• " + x for x in g["tests"][:5]) or "All selected materials have nearby microgravity data.") +
                "</span></div>", unsafe_allow_html=True)
    card = (f"FIRE-RISK CARD: {module}\nAtmosphere: {o2:.0%} O2 at {p_psi:.1f} psi (O2 partial pressure {flags['pp_o2_psi']:.1f} psi)\n"
            f"NFPA 99B increased burning: {'YES' if flags['increased_burning'] else 'no'} (limit {flags['limit_pct']:.1f}% O2)\n"
            f"Evidence readiness: {ready:.0%} of materials have microgravity fire data here\n\n"
            + "\n".join(f"- {r.item}: {r.label.split(' ', 1)[1]} ({r.n} tests; proxy {r.proxy})" for r in df.itertuples())
            + f"\n\nDetection: {g['detection']}\nSuppression: {g['suppression']}\nTests to run:\n"
            + "\n".join("- " + x for x in g["tests"]) + "\n\nSource: NASA PSI microgravity combustion data (psi.nasa.gov). "
            "Generated by the Flame in Freefall dashboard.")
    st.download_button("⬇️ Download this fire-risk card (.txt)", card, "fire_risk_card.txt", type="primary")
    st.caption("The traffic lights show whether evidence exists, not whether a material is safe. The pre-breathe index is a relative "
               "comparison (cabin nitrogen pressure ÷ suit pressure), not a medical criterion.")

# ================================================================== THE GAP
elif page == "🕳️ Data gap":
    st.markdown("## 🕳️ Where NASA's fire data runs out")
    c = st.columns(3)
    kpi(c[0], f"{len(known):,}", "tests with recorded oxygen and pressure", "blue")
    kpi(c[1], len(moon), "of them inside the Moon-base atmosphere", "orange")
    kpi(c[2], moon_solids, "solid cabin materials tested there", "red")
    fig = build_figure(tests)
    fig.add_scatter(x=[o2 * 100], y=[p_psi], mode="markers", name="Your selection",
                    marker=dict(symbol="x", size=16, color="#FF7A18", line=dict(width=2)))
    st.plotly_chart(fig, width="stretch")
    st.caption("Hollow dots: the PSI table didn't record O₂ or pressure, so ISS cabin air (≈21%, 14.7 psi) is assumed.")

    st.markdown("### 🧪 Recommended next experiments")
    st.markdown("Habitat-relevant conditions ranked by how little data exists. Solid materials are weighted "
                "highest because cabin fires start in fabrics, plastics and wiring.")
    gaps = gap_table(tests)
    for r in gaps.head(3).itertuples():
        st.markdown(f'<div class="card">🎯 <b>Test {r.fuel_type.lower()} at {r.o2_pct}% O₂ and {r.p_psi} psi</b> '
                    f'<span class="small">({r.condition})</span><br><span class="small">Data coverage {r.coverage:.0%} · '
                    f'closest existing test: {r.nearest}</span></div>', unsafe_allow_html=True)
    grp = st.radio("Show coverage for", list(PHASE_GROUPS), horizontal=True)
    o2s, ps = np.linspace(0.15, 1.0, 70), np.geomspace(3, 20, 60)
    z = coverage_grid(tests, PHASE_GROUPS[grp], o2s, ps)
    hm = go.Figure(go.Heatmap(x=o2s * 100, y=ps, z=z, colorscale=[[0, "#3a0d12"], [0.5, "#FF7A18"], [1, "#3DDC84"]],
                              zmin=0, zmax=1, colorbar=dict(title="coverage"),
                              hovertemplate="%{x:.0f}% O₂, %{y:.1f} psi<br>coverage %{z:.2f}<extra></extra>"))
    for name, (ho2, hp) in {"ISS cabin": (0.21, 1.0), "Moon base / rover / Mars": (0.30, 10.2 / PSI),
                            "Spacesuit": (1.0, 4.3 / PSI)}.items():
        hm.add_scatter(x=[ho2 * 100], y=[hp * PSI], mode="markers+text", text=[name],
                       textposition="top center", marker=dict(symbol="star", size=14, color="#FFD166"),
                       textfont=dict(color="#FFD166"), showlegend=False)
    hm.update_layout(title=f"How well {grp.lower()} are covered (green = tested nearby, dark = no data)",
                     xaxis_title="Oxygen (%)", yaxis=dict(title="Pressure (psi)", type="log"))
    st.plotly_chart(dark(hm, 460), width="stretch")
    with st.expander("Full gap table"):
        st.dataframe(gaps, hide_index=True, width="stretch")

    st.markdown("### 🧱 Will my material burn here?")
    fam = st.selectbox("Cabin material", list(MATERIALS))
    mt = material_tests(tests, fam)
    status, msg = material_verdict(mt, o2, p_psi)
    (st.success if status == "covered" else st.error if status == "gap" else st.warning)(
        f"**{fam} at {o2:.0%} O₂ / {p_psi:.1f} psi:** {msg}")
    if len(mt):
        st.dataframe(mt.assign(o2_pct=(mt.o2_fraction * 100).round(1), p_psi=(mt.pressure_atm * PSI).round(1))
                     [["investigation", "psi_id", "fuel", "o2_pct", "p_psi", "flow_cm_s", "o2_source"]]
                     .rename(columns={"o2_pct": "O₂ (%)", "p_psi": "pressure (psi)", "flow_cm_s": "air flow (cm/s)",
                                      "o2_source": "O₂ value"}),
                     hide_index=True, width="stretch", height=230)
    st.info("**No partial-gravity data exists in PSI yet.** Moon (1/6 g) and Mars (3/8 g) fires fall between the microgravity "
            "and Earth results. Upcoming: Lunar-g Combustion Investigation (LUCI) and Flammability of Materials on the Moon (FM2). "
            "ACME E-FIELD Flames used an electric field to mimic partial gravity.")

# ================================================================== TRADE-OFF
elif page == "⚖️ Trade-off":
    st.markdown("## ⚖️ The atmosphere trade-off")
    st.markdown("Before a spacewalk, astronauts breathe pure oxygen to flush nitrogen from their blood and avoid the bends. "
                "**Lower cabin pressure with more oxygen shortens that wait**, which is why the Moon base is planned at ≥30% O₂ "
                "and ≤10.2 psi. But more oxygen also makes fires worse, and that is exactly where NASA's fire data runs out.")
    grp = st.radio("Colour by fire-data coverage of", list(PHASE_GROUPS), horizontal=True, key="trade_grp")
    habs = {"ISS": (0.21, 14.7), "Moon base": (0.30, 10.2), "8.2 psi / 34%": (0.34, 8.2), "Spacesuit": (1.0, 4.3)}
    st.plotly_chart(dark(tradeoff_figure(tests, coverage_grid, PHASE_GROUPS[grp], o2, p_psi, habs), 620), width="stretch")
    st.markdown("""
**How to read it:**
- **Dotted contours:** pre-breathe index (cabin nitrogen pressure ÷ 4.3 psi suit pressure). Lower means a shorter pre-breathe. ISS air is 2.7; the Moon base 1.66.
- **Orange line (NFPA 99B):** to the right of it is an *atmosphere of increased burning*.
- **Green dashed line:** the same oxygen pressure you breathe at sea level (3.1 psi), so these atmospheres all feel like normal air to breathe.
- **Colour:** how much microgravity fire data exists nearby (yellow = well tested, dark = none).""")
    rows = []
    for name, (ho2, hp) in habs.items():
        fl = atmosphere_flags(ho2, hp)
        cov = {k: round(coverage(tests, ho2, hp, v)[0], 2) for k, v in PHASE_GROUPS.items()}
        rows.append({"Atmosphere": name, "O₂ (%)": round(ho2 * 100), "Pressure (psi)": hp,
                     "Pre-breathe index": round(fl["prebreathe_index"], 2),
                     "Increased burning (NFPA 99B)": "yes" if fl["increased_burning"] else "no",
                     **{f"Data: {k.split(' (')[0]}": v for k, v in cov.items()}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.info("**The insight:** below about 11.8 psi, every atmosphere that keeps normal breathing oxygen (green line) falls "
            "in the NFPA *increased burning* zone (right of the orange line), and that is where PSI has no microgravity data "
            "for solid cabin materials. New tests there, for example LUCI and FM2, would let designers choose a Moon-base "
            "atmosphere based on evidence instead of extrapolation.")
    st.caption("The NFPA 99B limit (O₂% > 23.45/√P in atm) comes from NFPA's Standard for Hypobaric Facilities, as cited by NASA "
               "White Sands Test Facility research. The pre-breathe index is a simplified relative measure, not a medical criterion.")

# ================================================================== PREDICT
elif page == "📈 Predict":
    st.markdown("## 📈 Predict a fire")
    st.markdown("A machine-learning model trained on **FLEX (PSI-69)**: 274 fuel-droplet fires burned on the ISS. "
                "Set the conditions; it predicts how fast the fuel burns and whether the flame goes out on its own.")
    a, b = st.columns([1, 1.6])
    with a:
        fuel = st.radio("Fuel", ["Methanol", "Heptane"], horizontal=True, help="Methanol is an alcohol; heptane is like petrol.")
        po2 = st.slider("Oxygen (%)", 10, 40, int(min(max(o2 * 100, 10), 40)),
                        help="FLEX tested 12–34%, so predictions above that are extrapolations.") / 100
        pp = st.slider("Pressure (psi)", 7.0, 45.0, float(min(max(p_psi, 7.0), 45.0)), 0.5) / PSI
        dil = st.selectbox("Diluent gas (what fills the rest of the air)", ["Nitrogen (like Earth air)", "CO₂ (extinguisher gas)",
                                                                          "Helium"])
        dfrac = 1 - po2
        co2, he = (dfrac * 0.8 if dil.startswith("CO") else 0.0), (dfrac if dil == "Helium" else 0.0)
        d0 = st.slider("Droplet size (mm)", 1.5, 4.5, 3.0, 0.1)
    r = fm.predict(fuel, po2, pp, co2, he, d0)
    lvl, msg = reliability(r["distance"])
    with b:
        c = st.columns(3)
        kpi(c[0], f"{r['burn_rate']['mid']:.2f}", f"burning rate, mm²/s<br>(likely {r['burn_rate']['low']:.2f}–{r['burn_rate']['high']:.2f})", "orange")
        kpi(c[1], f"{r['burn_time']['mid']:.0f} s", f"flame lifetime<br>(likely {r['burn_time']['low']:.0f}–{r['burn_time']['high']:.0f} s)", "blue")
        kpi(c[2], f"{r['p_self_ext']:.0%}", "chance the flame goes out on its own", "green" if r["p_self_ext"] > 0.5 else "red")
        (st.success if lvl == "high" else st.warning if lvl == "medium" else st.error)(
            f"{msg} · {r['n_similar']} real FLEX tests within ±3% O₂ / ±25% pressure.")
        grid = np.linspace(0.12, 0.40, 40)
        preds = [fm.predict(fuel, g, pp, co2 * (1 - g) / max(dfrac, 1e-9), he * (1 - g) / max(dfrac, 1e-9), d0) for g in grid]
        fig = go.Figure()
        fig.add_scatter(x=np.r_[grid, grid[::-1]] * 100, y=[p["burn_rate"]["high"] for p in preds] + [p["burn_rate"]["low"] for p in preds][::-1],
                        fill="toself", fillcolor="rgba(255,122,24,.18)", line=dict(width=0), name="likely range (10–90%)")
        fig.add_scatter(x=grid * 100, y=[p["burn_rate"]["mid"] for p in preds], line=dict(color="#FF7A18", width=3), name="prediction")
        real = fm.df[(fm.df.fuel == fuel) & (np.abs(np.log(fm.df.p_atm / pp)) < 0.3)]
        fig.add_scatter(x=real.o2 * 100, y=real.burn_rate, mode="markers", name="real FLEX tests (similar pressure)",
                        marker=dict(color="#5AA9FF", size=7, opacity=.7))
        fig.add_vrect(x0=34, x1=40, fillcolor="rgba(159,176,207,.10)", line_width=0,
                      annotation_text="beyond FLEX data", annotation_position="bottom right")
        fig.add_vline(x=21, line_dash="dot", annotation_text="Earth/ISS air", line_color="#9fb0cf")
        fig.add_vline(x=30, line_dash="dot", annotation_text="Moon base", line_color="#FFD166")
        fig.update_layout(title=f"{fuel} droplet burning rate vs oxygen at {pp * PSI:.1f} psi",
                          xaxis_title="Oxygen (%)", yaxis_title="Burning rate (mm²/s)", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(dark(fig, 400), width="stretch")
    m = fm.metrics
    with st.expander("🔍 How good is this model? (validation)"):
        st.markdown(f"""
Gradient-boosted trees (scikit-learn), checked with **5-fold cross-validation** on real FLEX tests (each test predicted by a model that never saw it):

| What it predicts | Tests | Accuracy |
|---|---|---|
| Burning rate | {m['burn_rate']['n']} | R² = {m['burn_rate']['r2']:.2f}, typical error ±{m['burn_rate']['mae']:.2f} mm²/s (median value {m['burn_rate']['median']:.2f}) |
| Flame lifetime | {m['burn_time']['n']} | R² = {m['burn_time']['r2']:.2f}, typical error ±{m['burn_time']['mae']:.1f} s (median {m['burn_time']['median']:.0f} s) |
| Flame goes out on its own | {m['self_extinguished']['n']} | {m['self_extinguished']['accuracy']:.0%} correct (always guessing the most common outcome: {max(m['self_extinguished']['base_rate'], 1 - m['self_extinguished']['base_rate']):.0%}) |

Inputs: fuel, oxygen, pressure, CO₂ and helium fractions, and droplet size. The shaded band uses quantile regression (10th–90th percentile).
**Limits:** FLEX covers 12–34% O₂ and about 0.7–3 atm, and only methanol and heptane droplets, not solid cabin materials.
Higher oxygen means faster burning and fewer flames that go out on their own, which is why Moon-base air is a concern.
A gas-flame ignition model was also tried on ACME CFI-G data but dropped, because it did no better than guessing.""")

# ================================================================== EQUIPMENT
elif page == "🧯 Equipment":
    st.markdown("## 🧯 What the data says about fire-safety equipment")
    t1, t2, t3 = st.tabs(["🧯 Suppression", "🔔 Detection", "🛡️ Prevention"])
    with t1:
        s = suppression_summary(fm.df)
        fig = go.Figure()
        for dname, color in [("Nitrogen (air-like)", "#9fb0cf"), ("CO₂", "#FF7A18"), ("Helium", "#5AA9FF")]:
            g = s[s.diluent == dname]
            fig.add_bar(x=g.o2_band.astype(str), y=g.self_ext_pct, name=dname, marker_color=color,
                        text=[f"{v}% (n={n})" for v, n in zip(g.self_ext_pct, g.tests)], textposition="outside")
        fig.update_layout(barmode="group", title="Share of FLEX droplet flames that went out on their own",
                          xaxis_title="Oxygen in the chamber", yaxis=dict(title="% self-extinguished", range=[0, 115]))
        st.plotly_chart(dark(fig, 430), width="stretch")
        st.markdown("**Takeaway:** below 15% O₂, over 90% of flames went out on their own; above 25% O₂, only 8% did. "
                    "At 20–25% O₂, helium-diluted air put out 77% of flames vs about 50% for nitrogen or CO₂. "
                    "Lowering oxygen is the strongest suppressant, and it works against you in O₂-enriched Moon-base air. "
                    "*Source: FLEX, PSI-69.*")
    with t2:
        sr = pd.read_csv(DATA / "raw" / "PSI-101__SAME-R.csv", encoding="latin-1")
        g = sr.groupby("Sample Type")["Rt/R0"].median().sort_values()
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color="#FFB020",
                               text=[f"+{(v - 1) * 100:.0f}%" for v in g.values], textposition="outside"))
        fig.update_layout(title="How much smoke particles grow as smoke ages (SAME-R)",
                          xaxis=dict(title="Particle size after aging ÷ initial size", range=[1, 1.6]))
        st.plotly_chart(dark(fig, 360), width="stretch")
        st.markdown("**Takeaway:** in microgravity, smoke particles grew 23–45% as the smoke aged, depending on the material "
                    "(Kapton and Teflon most). A detector far from the fire sees larger particles than one nearby, so "
                    "detectors should be tuned for the materials on board and for the larger particles that reach them. "
                    "*Sources: SAME-R (PSI-101), SAME (PSI-102). NASA's shuttle smoke detector that tells smoke from dust "
                    "is now used in mines.*")
    with t3:
        st.markdown("**Earth flammability tests don't reliably predict space fires** (Saffire-II, PSI-99, burned on a Cygnus cargo ship):")
        st.dataframe(SAFFIRE2, hide_index=True, width="stretch")
        st.markdown("**Takeaway:** thin silicone burned completely in Earth gravity but barely in microgravity, while "
                    "cotton-fiberglass fabric burned its full 29 cm length in microgravity. Material choices for "
                    "spacecraft need microgravity data, and that data barely exists above 22% O₂ (see *The data gap*). "
                    "*Related: BASS / BASS-II (PSI-26/25), 251 tests on PMMA, SIBAL and Nomex samples.*")

# ================================================================== EARTH
elif page == "🌍 Earth":
    st.markdown("## 🌍 Moon-base air already exists on Earth")
    st.markdown("Oxygen-rich air is not just a space problem. Hospitals, hyperbaric medicine and industry all work in "
                "**oxygen-enriched atmospheres**, where the same question applies: how do everyday materials burn when "
                "there's extra oxygen?")
    st.plotly_chart(dark(earth_figure(tests), 560), width="stretch")
    st.dataframe(EARTH.assign(**{"O₂ (%)": EARTH.o2_lo.astype(str) + "–" + EARTH.o2_hi.astype(str),
                                 "Pressure (psi)": EARTH.p_lo.astype(str) + "–" + EARTH.p_hi.astype(str)})
                 [["setting", "O₂ (%)", "Pressure (psi)", "note"]], hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    c1.markdown("""<div class="card"><b>🏥 Why it matters</b><br><span class="small">
    Hyperbaric chamber fires have killed patients and staff: a 1997 fire in a 1.8-atm multiplace chamber in Milan
    killed 11 people. A recent fatal monoplace fire in Troy, Michigan happened at 2.0 atm in ~100% O₂, with polyester and
    cotton linens inside. Those are the same kinds of fabrics as the cotton and Nomex samples NASA burned in
    microgravity.</span></div>""", unsafe_allow_html=True)
    c2.markdown("""<div class="card"><b>🔁 Space ↔ Earth</b><br><span class="small">
    PSI's flammability data for fabrics stops at ~22% O₂, just like the gap facing a Moon base. Tests that close the space gap
    (solids at 25–40% O₂) would also help hospitals and hyperbaric centres everywhere, including in Bahrain and the Gulf, choose
    safer materials. NASA spinoffs already did this once: shuttle smoke detectors, firefighter air tanks and protective
    clothing.</span></div>""", unsafe_allow_html=True)
    st.caption("Sources: NFPA 99 / OSHA definition of oxygen-enriched atmospheres (>23.5% O₂); hyperbaric practice of 100% O₂ "
               "at 2–3 atm (monoplace) and ≤23.5% chamber O₂ (multiplace); WHA International and the 2026 NBDHMT fire report. "
               "See the README for links.")

# ================================================================== EXPLORE & COMPARE
elif page == "🔬 Explore":
    st.markdown("## 🔬 Explore & compare")
    t1, t2 = st.tabs(["⚖️ Compare investigations", "📊 Data explorer"])
    with t1:
        pick = st.multiselect("Pick 2–3 investigations", ranked.id.tolist(), default=ranked.id.head(3).tolist(),
                              format_func=lambda i: f"{name_of[i]} ({i})", max_selections=3)
        if pick:
            def env(pid):
                t = tests[tests.psi_id == pid]
                f = lambda s, k=1, u="": f"{s.min() * k:.3g}–{s.max() * k:.3g}{u}" if s.notna().any() else "not recorded"
                return (f(t.o2_fraction, 100, " %"), f(t.pressure_atm, PSI, " psi"),
                        ", ".join(str(x) for x in t.fuel.value_counts().index[:4])) if len(t) else ("—", "—", "—")
            rows = {}
            for pid in pick:
                m, e = inv[inv.id == pid].iloc[0], enriched.get(pid, {})
                pillars, phase, _ = CURATED.get(pid, (["fundamental"], "gas", None))
                o2r, pr, fuels = env(pid)
                row = {"Fire-safety role": ", ".join(PILLAR_ICON.get(p, p) for p in pillars),
                       "Fuel type": PHASE_ICON.get(phase, phase), "Fuels tested": fuels,
                       "Where": f"{m.flight_program if isinstance(m.flight_program, str) else m.project_type}",
                       "Hardware": m.hardware if isinstance(m.hardware, str) else "—",
                       "O₂ tested": o2r, "Pressure tested": pr,
                       "Tests · papers": f"{(tests.psi_id == pid).sum()} · {m.n_publications}",
                       "Objective": (str(m.objective)[:280].rsplit(" ", 1)[0] + "…") if len(str(m.objective)) > 280 else str(m.objective)}
                for k, lbl in [("what_varied", "What varied (AI)"), ("measurements", "Measurements (AI)"),
                               ("key_findings", "Key findings (AI)")]:
                    if e.get(k):
                        row[lbl] = " • ".join(e[k]) if isinstance(e[k], list) else e[k]
                row["Cite (DOI)"] = m.doi
                rows[f"{name_of[pid]}"] = row
            st.table(pd.DataFrame(rows))
    with t2:
        sel = st.multiselect("Investigations", sorted(tests.investigation.unique()), default=["FLEX"])
        t = tests[tests.investigation.isin(sel)] if sel else tests
        labels = {"o2_fraction": "Oxygen (fraction)", "pressure_atm": "Pressure (atm)", "flow_cm_s": "Air flow (cm/s)",
                  "burn_time_s": "Burn time (s)"}
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X axis", list(labels), 0, format_func=labels.get)
        y = c2.selectbox("Y axis", list(labels), 3, format_func=labels.get)
        color = c3.selectbox("Colour by", ["fuel", "diluent", "outcome", "investigation", "fuel_phase"])
        tt = t.dropna(subset=[x, y])
        fig = go.Figure()
        for k, g in tt.groupby(color):
            fig.add_scatter(x=g[x], y=g[y], mode="markers", name=str(k), text=g.test_id, marker=dict(size=8, opacity=.8))
        fig.update_layout(xaxis_title=labels[x], yaxis_title=labels[y])
        st.plotly_chart(dark(fig, 440), width="stretch")
        st.dataframe(t, width="stretch", hide_index=True, height=260)
        st.download_button("⬇️ Download these tests (CSV)", t.to_csv(index=False), "psi_tests_filtered.csv")


# ================================================================== ASK
elif page == "💬 Ask":
    st.markdown("## 💬 Ask the data")
    st.caption("Answers come only from NASA PSI data and cite investigations. If the data doesn't cover your question, it says so.")

    def docs():
        out = []
        for r in inv.fillna("").itertuples():
            t = tests[tests.psi_id == r.id]
            env = (f"Tests: {len(t)}; O2 {t.o2_fraction.min():.2f}-{t.o2_fraction.max():.2f}; pressure "
                   f"{t.pressure_atm.min():.2f}-{t.pressure_atm.max():.2f} atm; fuels: {', '.join(str(x) for x in t.fuel.dropna().unique()[:8])}"
                   if len(t) and t.o2_fraction.notna().any() else "No O2/pressure test table.")
            out.append(f"[{r.id} | {name_of[r.id]} | DOI {r.doi}]\n{r.title}\nObjective: {r.objective}\n"
                       f"Impacts: {r.research_impacts}\n{env}\nAI attributes: {json.dumps(enriched.get(r.id, {}))}")
        return out

    def retrieve(q, k=5):
        from sklearn.feature_extraction.text import TfidfVectorizer
        D = docs()
        v = TfidfVectorizer(stop_words="english").fit(D + [q])
        sims = (v.transform(D) @ v.transform([q]).T).toarray().ravel()
        return [D[i] for i in sims.argsort()[::-1][:k]]

    examples = ["Which experiments are most relevant to a Moon base fire?",
                "What makes a droplet flame go out on its own in microgravity?",
                "Has any solid material been tested above 30% oxygen?",
                "How do smoke particles behave in microgravity, and what does that mean for detectors?"]
    ec = st.columns(2)
    for i, ex in enumerate(examples):
        ec[i % 2].button(ex, key=f"ex{i}", on_click=lambda e=ex: st.session_state.update(ask=e), width="stretch")
    q = st.chat_input("Ask about fire safety in space…") or st.session_state.pop("ask", None)
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
                st.warning("AI answers are switched off (no API key). Here are the most relevant investigations:")
                for c in ctx:
                    pid = c[1:].split(" |")[0]
                    st.markdown(f"**[{name_of.get(pid, pid)} ({pid})]({psi_url(pid)})**: {c.split('Objective: ')[1][:300]}…")
            else:
                import anthropic
                with st.spinner("Reading the investigations…"):
                    msg = anthropic.Anthropic(api_key=key).messages.create(
                        model=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"), max_tokens=900,
                        system=("You are a spacecraft fire-safety research assistant. Answer ONLY from the context, in plain "
                                "English for engineers and students. Cite every claim as [PSI-xx]. If the context does not "
                                "cover the question, say so plainly and name the gap. Keep it under 200 words. "
                                f"User habitat: {o2:.0%} O2 at {p_psi:.1f} psi."),
                        messages=[{"role": "user", "content": "CONTEXT:\n\n" + "\n\n".join(ctx) + f"\n\nQUESTION: {q}"}])
                st.markdown(msg.content[0].text)
                st.caption("AI-generated answer (Claude, Anthropic), grounded in the sources below. Check them before relying on it.")
            with st.expander("Sources used"):
                for c in ctx:
                    pid = c[1:].split(" |")[0]
                    st.markdown(f"- [{pid} · {name_of.get(pid, '')}]({psi_url(pid)})")

# ================================================================== ABOUT
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About Flame in Freefall")
    st.markdown(f"""
**Challenge:** NASA Space Apps 2026, *Flame in Freefall: SI-Powered Fire Safety Insights from Microgravity Combustion Data*.

**Data:** NASA Physical Sciences Informatics (PSI), all {len(inv)} combustion-science investigations, harvested from the
public PSI API on 14 Nov 2026 (CC0 licence). {len(tests):,} tests from 16 experimental tables were converted to one format (oxygen,
pressure, flow, fuel, outcome). Raw videos (≈11 TB) were not used.

**Assumptions:** where a PSI table doesn't record oxygen or pressure (BASS, SPICE, SLICE, SAME, Saffire), ISS cabin air
(≈21% O₂, 14.7 psi) is assumed and marked. Habitat atmospheres are from the challenge statement; Mars is assumed to match the Moon base.

**Methods:**
- *Relevance score:* weighted mix of fire-safety focus, closeness of tested conditions, fuel relevance and data richness.
- *Coverage / gaps:* distance from a condition to the nearest real test, using oxygen and log(pressure).
- *Prediction:* gradient-boosted trees on FLEX droplet fires, validated by 5-fold cross-validation (see *Predict a fire*).
""")
    st.markdown("#### 📖 Glossary")
    st.markdown("""
| Term | Meaning |
|---|---|
| Microgravity (µg) | Near-weightlessness, as on the ISS. No buoyancy, so hot gases don't rise. |
| Diluent | The non-oxygen gas in the air (nitrogen on Earth). CO₂ and helium were tested as fire suppressants. |
| Burning rate | How fast a fuel droplet shrinks as it burns (mm²/s). |
| Self-extinguished | The flame went out on its own before the fuel was used up. |
| Cool flame | A low-temperature flame that can keep burning after the visible flame goes out. |
| PSI ID / DOI | NASA's dataset identifier and the permanent citation link for each investigation. |
""")
    st.markdown("#### 🌍 Benefits on Earth")
    st.markdown("Fine-water-mist extinguishers developed for the ISS · a shuttle-era smoke detector that tells smoke from "
                "dust (now used in mining) · firefighter air tanks and protective clothing based on spacesuit technology. "
                "[Fire detector](https://spinoff.nasa.gov/node/9407) · [Air tanks](https://spinoff.nasa.gov/node/9582) · "
                "[Fire attire](https://spinoff.nasa.gov/node/9918)")
    st.markdown("#### 📚 Sources beyond PSI")
    st.markdown("""
- Oxygen-enriched atmosphere definitions (NFPA 99, OSHA, >23.5% O₂): [WHA International](https://wha-international.com/what-is-an-oxygen-enriched-atmosphere/)
- NFPA 99B "atmosphere of increased burning" (O₂% > 23.45/√P): [Beeson, NASA White Sands, ASTM STP](https://store.astm.org/stp12040s.html)
- Hyperbaric chamber practice and fires: [WHA International](https://wha-international.com/?p=1962) ·
  [NBDHMT fire report (2026)](https://nbdhmt.org/wp-content/uploads/2026/03/Catastrophic-Hyperbaric-Chamber-Fire-Report-dated-March-27-2026.pdf) ·
  [Baromedical, multiplace systems](https://baromedical.com/_files/ugd/e43c3a_e36f07ddab924991884056bf2a67bf24.pdf)
- Habitat atmospheres and pre-breathe context: the Flame in Freefall challenge statement.
""")
    st.markdown("#### 🤖 Use of AI (Super Intelligence)")
    st.markdown("""
- **Code:** parts of this dashboard's code were written with the help of Claude (Anthropic) and reviewed by the team.
- **AI summaries and chat answers** are generated by Claude from PSI text and are labelled in the app.
- **Habitat briefings and PDF reports** are generated by fixed rules from the data, not by an AI model.
- No AI-generated images. The Space Apps logo and the BASS flame photo are official, unmodified files loaded from the
  Space Apps website (spaceappschallenge.org).
""")

st.markdown("---")
st.caption("Flame in Freefall · NASA Space Apps 2026 · Data: NASA Physical Sciences Informatics (CC0). Cite each investigation "
           "by its DOI · Built with AI assistance (see About).")
