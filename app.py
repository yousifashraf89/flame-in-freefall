"""
Flame in Freefall: fire-safety insights for crewed exploration from NASA PSI microgravity combustion data.
Guided flow: choose a location -> choose a task -> focused page.
Run:  pip install -r requirements.txt  &&  streamlit run app.py
Built with AI assistance (Claude, Anthropic); see the About page.
"""
import json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from coverage_map import build_figure
from designer import (EARTH, INVENTORY, assess, atmosphere_flags, earth_figure, guidance, tradeoff_figure)
from features import (MATERIALS, PHASE_GROUPS, SAFFIRE2, briefing, coverage, coverage_grid, gap_table, material_tests,
                      material_verdict, pdf_report, suppression_summary)
from models import REJECTED, FlexModel, GasFlameModel, PooledModel, SmokeModel, raw_path, reliability
from scoring import CURATED, SHORT_NAMES, WEIGHTS, score

DATA = Path(__file__).parent / "data"
PSI = 14.696
LOGO = "https://assets.spaceappschallenge.org/media/images/Space_Apps_Logo_Color_and_White.width-440.jpegquality-60.png"
BASS_PHOTO = "https://assets.spaceappschallenge.org/media/images/Flame_in_Freefall2_4.3.2e16d0ba.fill-785x660.jpg"
st.set_page_config(page_title="Flame in Freefall", page_icon="🔥", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ style (light, high contrast, Space Apps palette)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@600;800&family=Overpass:wght@400;600;700&display=swap');
.stApp {background:#F5F7FB; color:#151622;}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none;}
.block-container {padding-top: 2.6rem; max-width: 1180px;}
.stMarkdown p, .stMarkdown li, .stMarkdown td {font-family:'Overpass',sans-serif; font-size:1.06rem; line-height:1.55; color:#151622;}
h1, h2, h3, h4 {font-family:'Fira Sans',sans-serif !important; color:#1B2B85 !important; font-weight:800 !important;}
.hero {background: linear-gradient(135deg, #0050D8 0%, #1B2B85 70%); border-radius: 20px; padding: 26px 30px;
       display:flex; gap:24px; align-items:center; margin-bottom: 22px;}
.hero .t {flex:1.7;} .hero .i {flex:1; text-align:right;}
.hero h1 {color:#fff !important; font-size:2.6rem; margin:0;} .hero p {color:#E8EEFF !important; font-size:1.12rem; margin:.4rem 0 0 0;}
.hero .tag {color:#E7F434; font-weight:700; letter-spacing:.06em; font-size:.82rem; text-transform:uppercase;}
.hero img {max-height:190px; max-width:100%; border-radius:14px;}
.step {font-family:'Fira Sans',sans-serif; font-weight:800; color:#1B2B85; font-size:1.35rem; margin:6px 0 10px 0;}
.step span {background:#FF5804; color:#fff; border-radius:999px; padding:2px 11px; margin-right:8px;}
.loc {background:#fff; border:2px solid #DFE1E2; border-radius:16px; padding:14px 16px 8px 16px; min-height:178px;}
.loc.sel {border-color:#FF5804; box-shadow:0 0 0 3px rgba(255,88,4,.15);}
.loc .ic {font-size:2rem;} .loc .nm {font-weight:800; font-size:1.08rem; color:#151622;} .loc .wh {color:#475569; font-size:.92rem;}
.chip {display:inline-block; background:#EEF2FF; color:#1B2B85; border-radius:999px; padding:2px 9px; font-size:.82rem;
       font-weight:600; margin:6px 4px 0 0;}
.chip.warn {background:#FFF1E8; color:#C2410C;}
.task {background:#fff; border:1px solid #DFE1E2; border-radius:16px; padding:16px 18px; min-height:165px;}
.task .ic {font-size:1.9rem;} .task .tt {font-weight:800; font-size:1.12rem; color:#151622;} .task .ds {color:#475569; font-size:.95rem;}
.kpi {background:#fff; border:1px solid #DFE1E2; border-radius:16px; padding:16px 18px; height:100%;}
.kpi .v {font-family:'Fira Sans',sans-serif; font-size:2.2rem; font-weight:800; line-height:1.1;}
.kpi .l {color:#334155; font-size:.95rem; margin-top:4px;}
.kpi.red .v {color:#E52207;} .kpi.orange .v {color:#EA580C;} .kpi.blue .v {color:#0050D8;} .kpi.green .v {color:#15803D;}
.card {background:#fff; border:1px solid #DFE1E2; border-radius:14px; padding:14px 16px; margin-bottom:10px; color:#151622;}
.small {color:#475569; font-size:.92rem;}
.verdict {border-radius:14px; padding:16px 20px; font-size:1.12rem; margin:4px 0 16px 0; color:#151622;}
.verdict.red {background:#FEECEC; border-left:7px solid #E52207;} .verdict.yellow {background:#FFF7E0; border-left:7px solid #EAB308;}
.verdict.green {background:#E9F8EE; border-left:7px solid #16A34A;}
.risk {background:#fff; border:1px solid #DFE1E2; border-radius:12px; padding:10px 14px; margin-bottom:8px;}
.risk.green {border-left:7px solid #16A34A;} .risk.yellow {border-left:7px solid #EAB308;}
.risk.orange {border-left:7px solid #EA580C;} .risk.red {border-left:7px solid #E52207;}
.bar {height:10px; background:#E5E7EB; border-radius:99px; overflow:hidden;} .bar > div {height:10px; background:#FF5804;}
div.stButton > button {border-radius:12px; font-weight:700; padding:.55rem 1rem;}
div.stButton > button[kind="primary"] {background:#FF5804; border-color:#FF5804; color:#fff;}
div.stDownloadButton > button {border-radius:12px; font-weight:700; background:#1B2B85; color:#fff; border-color:#1B2B85;}
a {color:#0050D8 !important; font-weight:600;}
.help {background:#EEF2FF; border-radius:12px; padding:10px 14px; color:#1B2B85; font-size:1rem; margin:0 0 14px 0;}
.lesson {background:#fff; border:1px solid #DFE1E2; border-left:6px solid #0050D8; border-radius:12px; padding:12px 16px; margin-bottom:10px;}
</style>""", unsafe_allow_html=True)

LOCATIONS = {  # atmospheres: challenge statement unless marked "assumed"
    "iss": dict(icon="🛰️", name="International Space Station", where="Low Earth orbit", g="Microgravity",
                o2=0.21, p=14.7, src="ISS cabin air"),
    "gateway": dict(icon="🌘", name="Lunar Gateway", where="Space station orbiting the Moon", g="Microgravity",
                    o2=0.21, p=14.7, src="assumed ISS-like air"),
    "moonbase": dict(icon="🌕", name="Moon base", where="Lunar surface, south-pole region", g="1/6 g",
                     o2=0.30, p=10.2, src="planned (challenge statement)"),
    "rover": dict(icon="🚙", name="Pressurized lunar rover", where="Driving across the Moon", g="1/6 g",
                  o2=0.30, p=10.2, src="planned (challenge statement)"),
    "transit": dict(icon="🚀", name="Mars transit ship", where="Deep space, Earth → Mars", g="Microgravity",
                    o2=0.21, p=14.7, src="assumed ISS-like air"),
    "marsbase": dict(icon="🔴", name="Mars base", where="Martian surface", g="3/8 g",
                     o2=0.30, p=10.2, src="assumed same as Moon base"),
    "suit": dict(icon="👨‍🚀", name="Spacesuit", where="Spacewalks on the Moon or Mars", g="Varies",
                 o2=1.00, p=4.3, src="spacesuit (challenge statement)"),
}
TASKS = [
    ("risk", "🔥", "How risky is fire here?", "A plain-language risk report: what NASA tested, what's missing, the most relevant experiments, a PDF."),
    ("designer", "🏗️", "Check my module's materials", "Pick what's on board and get a fire-risk card with a traffic light for each material."),
    ("predict", "📈", "Predict a fire", "Machine-learning models from FLEX, CFI-G and SAME-R: liquid, gas and smoke fires."),
    ("gaps", "🕳️", "Where is NASA's data missing?", "Coverage maps, the tests NASA should run next, and 'will my material burn here?'"),
    ("equipment", "🧯", "What protects the crew?", "What the data says about putting fires out, detecting smoke and choosing materials."),
    ("atmosphere", "⚖️", "Choose the best atmosphere", "Shorter wait before spacewalks vs higher fire risk, in one chart."),
    ("lessons", "📜", "What NASA has learned", "Past spacecraft fires, key lessons from the ISS, and the experiments coming next."),
    ("data", "📚", "Explore NASA's data", "All 24 investigations in one library. Compare them and download the clean data."),
    ("ask", "💬", "Ask a question", "Ask in plain English. Answers cite NASA investigations and reports."),
]
PILLAR_ICON = {"prevention": "🛡️ Prevention", "detection": "🔔 Detection", "suppression": "🧯 Suppression",
               "fundamental": "🔬 Fundamental"}
PHASE_ICON = {"solid": "🧱 Solid", "liquid": "💧 Liquid", "gas": "💨 Gas", "gas (solid emulator)": "💨 Gas (mimics a solid)",
              "dust": "🌫️ Dust"}


# ------------------------------------------------------------------ data & models
@st.cache_data
def load():
    inv = pd.read_csv(DATA / "investigations.csv")
    tests = pd.read_csv(DATA / "tests_normalized.csv")
    p = DATA / "investigations_enriched.json"
    return inv, tests, (json.load(open(p)) if p.exists() else {})


@st.cache_resource
def models():
    return FlexModel(), GasFlameModel(), SmokeModel(), PooledModel()


inv, tests, enriched = load()
fm, gm, sm, pm = models()
name_of = {r.id: SHORT_NAMES.get(r.id) or (r.acronym if isinstance(r.acronym, str) else r.title[:45]) for r in inv.itertuples()}
psi_url = lambda pid: f"https://psi.nasa.gov/physci/repo/data/investigations/{pid}"
known = tests.dropna(subset=["o2_fraction", "pressure_atm"])

ss = st.session_state
ss.setdefault("view", "home")
ss.setdefault("loc", None)
VIEWS_ALL = ["home", "risk", "designer", "predict", "gaps", "equipment", "atmosphere", "lessons", "data", "ask", "earth", "about"]
if not ss.get("_qp_loaded"):                      # open a shared link straight on a page + location
    ss._qp_loaded = True
    _qp = st.query_params
    if _qp.get("loc") in LOCATIONS:
        ss.loc = _qp["loc"]
    if _qp.get("view") in VIEWS_ALL and (ss.loc or _qp.get("view") in ("home", "earth", "about")):
        ss.view = _qp["view"]


def share_url():
    q = f"?view={ss.view}" + (f"&loc={ss.loc}" if ss.loc else "")
    try:
        base = str(st.context.url).split("?")[0]
    except Exception:
        try:
            base = f"https://{st.context.headers.get('Host', '')}/"
        except Exception:
            base = ""
    return base + q


PRESETS = {"Balanced": WEIGHTS,
           "Habitat designer": dict(pillar=0.25, envelope=0.45, fuel=0.20, data=0.10),
           "Materials engineer": dict(pillar=0.30, envelope=0.20, fuel=0.40, data=0.10),
           "Modeller": dict(pillar=0.20, envelope=0.20, fuel=0.10, data=0.50)}


def set_preset(p):
    ss.update({f"w_{k}": float(v) for k, v in PRESETS[p].items()})


def go_view(v):
    ss.view = v


def pick_loc(k):
    ss.loc = k
    ss.pop("o2_adj", None)
    ss.pop("p_adj", None)
    ss.pop(f"o2_pct_{k}", None)
    ss.pop(f"p_sl_{k}", None)


def explain(text):
    st.markdown(f'<div class="help">💡 {text}</div>', unsafe_allow_html=True)


EXTRA = DATA / "extra"


@st.cache_data
def extra():
    rd = lambda n: pd.read_csv(EXTRA / n) if (EXTRA / n).exists() else pd.DataFrame()
    kb = json.load(open(DATA / "knowledge.json")) if (DATA / "knowledge.json").exists() else []
    gc = json.load(open(EXTRA / "guide_chunks.json")) if (EXTRA / "guide_chunks.json").exists() else []
    return rd("bass_as_run.csv"), rd("bass2_flow_runs.csv"), rd("cool_flame_extinction.csv"), rd("flex_droplet_traces.csv"), kb, gc


def kpi(col, value, label, tone="blue"):
    col.markdown(f'<div class="kpi {tone}"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


def light(fig, h=440):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", height=h, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(size=13, color="#151622"))
    return fig


def current():
    loc = LOCATIONS[ss.loc or "moonbase"]
    o2 = ss.get("o2_adj", loc["o2"])
    p_psi = ss.get("p_adj", loc["p"])
    return loc, o2, p_psi


def top_bar(title):
    loc, o2, p_psi = current()
    a, b, d, c = st.columns([1, 4, 1, 1.3])
    a.button("← Back", on_click=go_view, args=("home",), width="stretch")
    with d.popover("🔗 Share", width="stretch"):
        st.caption("Copy this link: it opens this exact page and location.")
        st.code(share_url(), language=None)
    b.markdown(f'<div style="padding-top:6px"><span class="chip">{loc["icon"]} {loc["name"]}</span>'
               f'<span class="chip">{o2:.0%} O₂ · {p_psi:.1f} psi</span><span class="chip">{loc["g"]}</span>'
               + ('<span class="chip warn">no partial-gravity data in PSI</span>' if "g" in loc["g"] and "/" in loc["g"] else "")
               + "</div>", unsafe_allow_html=True)
    c.button("Change location", on_click=lambda: (go_view("home"), ss.update(loc=None)), width="stretch")
    st.markdown(f"## {title}")


bass_ar, bass2, coolf, traces, KB, GUIDE = extra()
kb = {k["id"]: k for k in KB}

# ================================================================== HOME
if ss.view == "home":
    st.markdown(f"""<div class="hero"><div class="t"><div class="tag">NASA Space Apps 2026 · Flame in Freefall</div>
    <h1>🔥 Flame in Freefall</h1><p>Fire behaves differently in space. We turned decades of NASA microgravity fire experiments
    into simple answers: <b>how risky is fire where your crew lives, and what to do about it.</b></p></div>
    <div class="i"><img src="{BASS_PHOTO}" alt="Flame from a NASA BASS experiment">
    <div style="color:#C7D2FE;font-size:.75rem">A flame in microgravity · Image: NASA (BASS)</div></div></div>""",
                unsafe_allow_html=True)

    _moon = known[(known.o2_fraction >= 0.30) & (known.pressure_atm * PSI <= 10.2)]
    k = st.columns(4)
    kpi(k[0], len(inv), "NASA combustion investigations, combined", "blue")
    kpi(k[1], f"{len(tests):,}", "fire tests cleaned into one table", "green")
    kpi(k[2], "≥30%", "oxygen planned for Moon-base air (Earth: 21%)", "orange")
    kpi(k[3], int(_moon.fuel_phase.str.startswith("solid").sum()), f"solid-material tests at Moon-base conditions (of {len(_moon)})", "red")
    st.write("")
    st.markdown('<div class="step"><span>1</span>Where is your crew?</div>', unsafe_allow_html=True)
    keys = list(LOCATIONS)
    for row in (keys[:4], keys[4:]):
        cols = st.columns(4)
        for c, k in zip(cols, row):
            L = LOCATIONS[k]
            sel = ss.loc == k
            c.markdown(f'<div class="loc {"sel" if sel else ""}"><div class="ic">{L["icon"]}</div><div class="nm">{L["name"]}</div>'
                       f'<div class="wh">{L["where"]}</div><span class="chip">{L["g"]}</span>'
                       f'<span class="chip">{L["o2"]:.0%} O₂ · {L["p"]} psi</span></div>', unsafe_allow_html=True)
            c.button("✓ Selected" if sel else "Choose", key=f"loc_{k}", on_click=pick_loc, args=(k,),
                     type="primary" if sel else "secondary", width="stretch")

    if ss.loc:
        loc, o2, p_psi = current()
        st.caption(f"Atmosphere source: {loc['src']}.")
        with st.expander("⚙️ Adjust the atmosphere (optional)"):
            c1, c2 = st.columns(2)
            ko, kp = f"o2_pct_{ss.loc}", f"p_sl_{ss.loc}"
            c1.slider("Oxygen (%)", 15, 100, int(round(o2 * 100)), key=ko,
                      on_change=lambda: ss.update(o2_adj=ss[ko] / 100), help="Earth air is 21%.")
            c2.slider("Pressure (psi)", 3.0, 16.0, float(p_psi), 0.1, key=kp,
                      on_change=lambda: ss.update(p_adj=ss[kp]), help="Sea level is 14.7 psi.")
        st.markdown('<div class="step"><span>2</span>What do you want to know?</div>', unsafe_allow_html=True)
        for row in (TASKS[:3], TASKS[3:6], TASKS[6:]):
            cols = st.columns(3)
            for c, (k, ic, tt, ds) in zip(cols, row):
                c.markdown(f'<div class="task"><div class="ic">{ic}</div><div class="tt">{tt}</div><div class="ds">{ds}</div></div>',
                           unsafe_allow_html=True)
                c.button("Open →", key=f"task_{k}", on_click=go_view, args=(k,), type="primary", width="stretch")
    else:
        st.info("👆 Choose a location to continue.")
    st.write("")
    f1, f2, f3 = st.columns([1, 1, 2])
    f1.button("🌍 Why it matters on Earth", on_click=go_view, args=("earth",), width="stretch")
    f2.button("ℹ️ About, methods & sources", on_click=go_view, args=("about",), width="stretch")
    f3.markdown(f'<div class="small" style="padding-top:8px">Data: NASA Physical Sciences Informatics · {len(inv)} investigations · '
                f'{len(tests):,} tests · built for the NASA Space Apps Challenge 2026</div>', unsafe_allow_html=True)

# ================================================================== RISK REPORT
elif ss.view == "risk":
    loc, o2, p_psi = current()
    top_bar(f"🔥 How risky is fire at the {loc['name']}?")
    explain("Start here. The coloured box gives the verdict, the numbers show how much NASA data exists for this air, and the "
            "list shows which experiments matter most. Download the PDF to share it.")
    flags = atmosphere_flags(o2, p_psi)
    near = known[(known.o2_fraction.sub(o2).abs() <= 0.03) & ((known.pressure_atm * PSI / p_psi).between(0.8, 1.25))]
    solids_near = int(near.fuel_phase.str.startswith("solid").sum())
    if flags["increased_burning"] and solids_near == 0:
        v, msg = "red", (f"🔴 <b>High uncertainty.</b> This air ({o2:.0%} O₂ at {p_psi:.1f} psi) counts as an <b>atmosphere of increased "
                         f"burning</b> (NFPA 99B), and NASA has never burned a solid cabin material in microgravity at these conditions.")
    elif flags["oea"] or solids_near == 0:
        v, msg = "yellow", ("🟡 <b>Partly covered.</b> Some NASA tests are nearby, but not for every material. "
                            "Check your module's materials.")
    else:
        v, msg = "green", ("🟢 <b>Well studied.</b> NASA's microgravity experiments cover this atmosphere, including solid "
                           "materials like fabrics and plastics.")
    if "/" in loc["g"]:
        msg += (f" <br><span class='small'>Note: PSI has no {loc['g']} data yet. NASA's FM2 project notes that materials may burn at "
                "<b>lower</b> oxygen in lunar gravity than on Earth, so partial gravity could be the worst case.</span>")
    st.markdown(f'<div class="verdict {v}">{msg}</div>', unsafe_allow_html=True)
    c = st.columns(3)
    kpi(c[0], f"{len(near)}", f"NASA tests near this atmosphere (of {len(known):,})", "blue")
    kpi(c[1], solids_near, "of them on solid materials (fabrics, plastics)", "red" if solids_near == 0 else "green")
    kpi(c[2], f"{flags['pp_o2_psi']:.1f} psi", "oxygen partial pressure (Earth sea level: 3.1)", "orange")
    st.write("")
    fig = build_figure(tests)
    fig.add_scatter(x=[o2 * 100], y=[p_psi], mode="markers", name="This location",
                    marker=dict(symbol="x", size=18, color="#FF5804", line=dict(width=3)))
    st.plotly_chart(light(fig, 520), width="stretch")
    st.caption("Each dot is one NASA microgravity fire test. The orange × is your location. Hollow dots: oxygen or pressure not "
               "recorded, so cabin air is assumed.")

    w_raw = {k: ss.get(f"w_{k}", WEIGHTS[k]) for k in WEIGHTS}
    w = {k: v / (sum(w_raw.values()) or 1) for k, v in w_raw.items()}
    ranked = score(inv, tests, o2, p_psi / PSI, w)
    brief = briefing(loc["name"], o2, p_psi, ranked, tests, fm.predict("Methanol", min(max(o2, .12), .40), max(p_psi, 7) / PSI))
    st.markdown('<div class="card"><b>In short:</b> ' + re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", brief) +
                '<br><span class="small">Written automatically from the PSI data.</span></div>', unsafe_allow_html=True)
    st.markdown("### 🏆 The NASA experiments that matter most here")
    for r in ranked.head(5).itertuples():
        meta = inv[inv.id == r.id].iloc[0]
        line = enriched.get(r.id, {}).get("one_line_summary") or str(meta.objective).split(". ")[0]
        line = line if len(line) < 140 else line[:140].rsplit(" ", 1)[0] + "…"
        st.markdown(f'<div class="card"><b>{r.name}</b> <span class="small">({r.id}) · {", ".join(PILLAR_ICON.get(p, p) for p in r.pillars.split(", "))}'
                    f' · {PHASE_ICON.get(r.fuel_phase, r.fuel_phase)} · {r.n_tests} tests</span>'
                    f'<div class="bar" style="margin:8px 0"><div style="width:{r.score:.0f}%"></div></div>'
                    f'<span class="small">Relevance {r.score:.0f}/100 · {line} · <a href="{psi_url(r.id)}" target="_blank">open in PSI</a></span></div>',
                    unsafe_allow_html=True)

    st.markdown("### 🧯 What protects the crew")
    s = suppression_summary(fm.df)
    hi = s[s.o2_band.astype(str) == ">25%"].self_ext_pct.max()
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="card"><b>🧯 Put fires out</b><br>Above 25% O₂, only <b>{hi:.0f}%</b> of FLEX droplet flames went out on '
                'their own (over 90% below 15%). Lowering oxygen is the strongest suppressant.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><b>🔔 Detect smoke</b><br>In microgravity, smoke particles grew <b>23–45%</b> as smoke aged '
                '(SAME-R). Tune detectors for the materials on board.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><b>🛡️ Choose materials</b><br>Earth tests can mislead: thin silicone burned fully on Earth but '
                'barely in microgravity (Saffire-II). Test materials in microgravity.</div>', unsafe_allow_html=True)

    st.markdown("### 🧪 What NASA should test next")
    cov = {k: coverage(tests, o2, p_psi, v)[0] for k, v in PHASE_GROUPS.items()}
    for k, val in sorted(cov.items(), key=lambda x: x[1]):
        tone = "red" if val < 0.3 else "yellow" if val < 0.8 else "green"
        st.markdown(f'<div class="risk {tone}"><b>{k}</b> at {o2:.0%} O₂ / {p_psi:.1f} psi: data coverage <b>{val:.0%}</b>'
                    + (" → <b>recommended test</b>" if val < 0.8 else " → covered") + "</div>", unsafe_allow_html=True)
    gaps = gap_table(tests)
    st.download_button("📄 Download this report (PDF)", pdf_report(loc["name"], o2, p_psi, ranked, gaps, brief, fm.metrics),
                       "flame_in_freefall_report.pdf", "application/pdf")
    with st.expander("⚖️ How the relevance score works (and change the weights)"):
        st.markdown("Score = fire-safety focus + how close NASA's tests came to this atmosphere + fuel relevance "
                    "(solids count most) + amount of data. Default weights 35 / 30 / 20 / 15.")
        st.markdown("**Quick presets:** pick who you are, or move the sliders yourself.")
        pc = st.columns(4)
        for col, (pname, why) in zip(pc, [("Balanced", "our default"), ("Habitat designer", "closest atmosphere first"),
                                          ("Materials engineer", "solid materials first"), ("Modeller", "most data first")]):
            col.button(pname, on_click=set_preset, args=(pname,), key=f"pre_{pname}", help=why, width="stretch")
        wc = st.columns(4)
        for col, (k, lbl) in zip(wc, [("pillar", "Fire-safety focus"), ("envelope", "Closeness of tests"),
                                      ("fuel", "Fuel relevance"), ("data", "Data richness")]):
            ss.setdefault(f"w_{k}", WEIGHTS[k])
            col.slider(lbl, 0.0, 1.0, step=0.05, key=f"w_{k}")
        top = ranked.head(10)
        fig = go.Figure()
        for k, label, color in [("pillar", "Fire-safety focus", "#E52207"), ("envelope", "Closeness of tests", "#FF5804"),
                                ("fuel", "Fuel relevance", "#16A34A"), ("data", "Data richness", "#0050D8")]:
            fig.add_bar(y=top.name, x=top[f"pts_{k}"], name=label, orientation="h", marker_color=color)
        fig.update_layout(barmode="stack", yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(light(fig, 420), width="stretch")
        st.markdown("**All 24 investigations**")
        st.dataframe(ranked[["score", "name", "id", "pillars", "fuel_phase", "n_tests"]], hide_index=True, width="stretch")

# ================================================================== DESIGNER
elif ss.view == "designer":
    loc, o2, p_psi = current()
    top_bar(f"🏗️ Check the materials in your {loc['name'].lower()} module")
    explain("Name your module, tick what's on board, and read the traffic lights: they show whether NASA has ever burned "
            "that material in microgravity at this air.")
    n1, n2 = st.columns([3, 1])
    module = n1.text_input("Module name", f"{loc['name']} habitat module")
    crew = n2.number_input("Crew", 1, 8, 4)
    items = st.multiselect("What's on board?", list(INVENTORY), default=list(INVENTORY)[:6],
                           help="Each item is matched to the closest material NASA actually tested.")
    df, flags, fx, ready = assess(tests, items, o2, p_psi, fm)
    c = st.columns(3)
    kpi(c[0], f"{ready:.0%}", "of these materials have NASA fire data at this atmosphere",
        "green" if ready > 0.7 else "orange" if ready > 0.3 else "red")
    kpi(c[1], "Yes" if flags["increased_burning"] else "No", "'Atmosphere of increased burning' (NFPA 99B)?",
        "red" if flags["increased_burning"] else "green")
    kpi(c[2], f"{flags['prebreathe_index']:.2f}", "pre-breathe index (ISS 2.70; lower = shorter wait before spacewalks)", "blue")
    if flags["increased_burning"]:
        st.markdown(f'<div class="verdict red">🔥 <b>Atmosphere of increased burning</b> (NFPA 99B): {o2:.0%} O₂ is above the '
                    f'{flags["limit_pct"]:.1f}% limit for {p_psi:.1f} psi. Materials ignite more easily and burn faster than on Earth.</div>',
                    unsafe_allow_html=True)
    elif flags["oea"]:
        st.markdown('<div class="verdict yellow">⚠️ <b>Oxygen-enriched air</b> (above 23.5% O₂, NFPA 99 / OSHA).</div>', unsafe_allow_html=True)
    st.markdown(f"### 🪪 Fire-risk card: {module} · crew {crew}")
    st.caption("🟢 tested near these conditions · 🟡 tested only in Earth-like air · 🟠 only heated for smoke tests, never burned · "
               "🔴 never tested")
    cols = st.columns(2)
    for i, r in enumerate(df.itertuples()):
        cols[i % 2].markdown(f'<div class="risk {r.status}"><b>{r.item}</b><br>{r.label} '
                             f'<span class="small">· {r.n} tests</span><br><span class="small">{r.note}</span></div>',
                             unsafe_allow_html=True)
    g = guidance(df, flags, fx, o2)
    st.markdown("### What to do")
    lvl, msg_r = reliability(fx["distance"])
    st.markdown(f'<div class="card"><b>🔥 How a small fire would behave here</b> (FLEX model, methanol droplet): burns at '
                f'≈{fx["burn_rate"]["mid"]:.2f} mm²/s and goes out on its own {fx["p_self_ext"]:.0%} of the time. '
                f'<span class="small">{msg_r}</span></div>', unsafe_allow_html=True)
    a, b, c3 = st.columns(3)
    a.markdown(f'<div class="card"><b>🔔 Detection</b><br><span class="small">{g["detection"]}</span></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="card"><b>🧯 Suppression</b><br><span class="small">{g["suppression"]}</span></div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><b>🧪 Test before flight</b><br><span class="small">' +
                ("<br>".join("• " + x for x in g["tests"][:5]) or "All selected materials have nearby data.") + "</span></div>",
                unsafe_allow_html=True)
    card = (f"FIRE-RISK CARD: {module} at {loc['name']} ({o2:.0%} O2, {p_psi:.1f} psi), crew {crew}\nEvidence readiness: {ready:.0%}\n"
            + "\n".join(f"- {r.item}: {r.label.split(' ', 1)[1]} ({r.n} tests)" for r in df.itertuples())
            + f"\nDetection: {g['detection']}\nSuppression: {g['suppression']}\nTests to run:\n" + "\n".join("- " + x for x in g["tests"])
            + "\nSource: NASA PSI (psi.nasa.gov) via Flame in Freefall.")
    st.download_button("⬇️ Download the fire-risk card", card, "fire_risk_card.txt")
    st.caption("Traffic lights show whether evidence exists, not whether a material is safe.")

# ================================================================== PREDICT
elif ss.view == "predict":
    loc, o2, p_psi = current()
    top_bar("📈 Predict a fire")
    explain("Pick a tab for the kind of fire. Move the sliders; the big numbers update. A green box means the prediction is "
            "close to real tests, yellow means it's an estimate, red means you're outside NASA's tested range.")
    st.markdown("Four models trained on **three NASA experiments** (FLEX, ACME CFI-G and SAME-R; 515 tests in total). Each "
                "one says how sure it is and warns you when you leave the tested range.")
    t1, t2, t3, t4, t5 = st.tabs(["💧 Liquid fuel fire", "💨 Gas-leak flame", "🌫️ Smoke & detectors", "🔗 All fuels together",
                                  "✅ How accurate?"])
    with t1:
        st.markdown("**A burning drop of fuel** (e.g. a leaking solvent), from **FLEX: 274 droplet fires on the ISS**.")
        a, b = st.columns([1, 1.6])
        fuel = a.radio("Fuel", ["Methanol", "Heptane"], horizontal=True, help="Methanol is an alcohol; heptane is like petrol.")
        po2 = a.slider("Oxygen (%)", 10, 40, int(min(max(o2 * 100, 10), 40)), key="f_o2") / 100
        pp = a.slider("Pressure (psi)", 7.0, 45.0, float(min(max(p_psi, 7.0), 45.0)), 0.5, key="f_p") / PSI
        dil = a.selectbox("Rest of the air", ["Nitrogen (like Earth)", "CO₂ (extinguisher gas)", "Helium"])
        co2, he = ((1 - po2) * 0.8 if dil.startswith("CO") else 0.0), ((1 - po2) if dil == "Helium" else 0.0)
        d0 = a.slider("Droplet size (mm)", 1.5, 4.5, 3.0, 0.1, help="FLEX burned droplets about 1–5 mm across.")
        r = fm.predict(fuel, po2, pp, co2, he, d0)
        lvl, msg = reliability(r["distance"])
        with b:
            c = st.columns(3)
            kpi(c[0], f"{r['burn_rate']['mid']:.2f}", f"burning rate, mm²/s<br>(likely {r['burn_rate']['low']:.2f}–{r['burn_rate']['high']:.2f})", "orange")
            kpi(c[1], f"{r['burn_time']['mid']:.0f} s", f"flame lifetime<br>(likely {r['burn_time']['low']:.0f}–{r['burn_time']['high']:.0f} s)", "blue")
            kpi(c[2], f"{r['p_self_ext']:.0%}", "chance it goes out on its own", "green" if r["p_self_ext"] > .5 else "red")
            (st.success if lvl == "high" else st.warning if lvl == "medium" else st.error)(msg)
            grid = np.linspace(0.12, 0.40, 40)
            preds = [fm.predict(fuel, x, pp, co2 * (1 - x) / max(1 - po2, 1e-9), he * (1 - x) / max(1 - po2, 1e-9), d0) for x in grid]
            fig = go.Figure()
            fig.add_scatter(x=np.r_[grid, grid[::-1]] * 100, y=[q["burn_rate"]["high"] for q in preds] + [q["burn_rate"]["low"] for q in preds][::-1],
                            fill="toself", fillcolor="rgba(255,88,4,.15)", line=dict(width=0), name="likely range")
            fig.add_scatter(x=grid * 100, y=[q["burn_rate"]["mid"] for q in preds], line=dict(color="#FF5804", width=3), name="prediction")
            real = fm.df[(fm.df.fuel == fuel) & (np.abs(np.log(fm.df.p_atm / pp)) < 0.3)]
            fig.add_scatter(x=real.o2 * 100, y=real.burn_rate, mode="markers", name="real FLEX tests", marker=dict(color="#0050D8", size=7))
            fig.add_vrect(x0=34, x1=40, fillcolor="rgba(100,116,139,.12)", line_width=0, annotation_text="beyond FLEX data")
            fig.update_layout(title="Burning rate vs oxygen", xaxis_title="Oxygen (%)", yaxis_title="mm²/s", legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(light(fig, 360), width="stretch")
        st.markdown("#### 🔬 Inside the data: watch a droplet burn, and the hidden 'cool flame'")
        c1, c2 = st.columns(2)
        if len(traces):
            fig = go.Figure()
            for k, g_ in traces.groupby("test"):
                fig.add_scatter(x=g_.t_s, y=g_.d2_mm2, mode="markers+lines", name=k, marker=dict(size=4))
            fig.update_layout(title="A burning droplet shrinks steadily (FLEX raw data)", xaxis_title="time (s)",
                              yaxis_title="droplet diameter² (mm²)", legend=dict(orientation="h", y=-0.3))
            c1.plotly_chart(light(fig, 330), width="stretch")
            c1.caption("The slope of this line is the burning rate the model predicts. Source: PSI-69 raw droplet data.")
        if len(coolf):
            fig = go.Figure()
            for k, g_ in coolf.groupby("kind"):
                fig.add_scatter(x=g_.pressure_atm, y=g_.extinction_diameter_mm, mode="markers+lines", name=k,
                                marker=dict(size=9, color="#E52207" if "hot" in k else "#0050D8"))
            fig.update_layout(title="Visible flame goes out first; the cool flame keeps going", xaxis_title="pressure (atm)",
                              yaxis_title="droplet size when flame went out (mm)", legend=dict(orientation="h", y=-0.3))
            c2.plotly_chart(light(fig, 330), width="stretch")
            c2.caption("n-dodecane, ~4 mm droplets in air. The visible flame dies at ~3.5 mm, but an invisible cool flame "
                       "can burn on until the droplet is <1 mm. Source: PSI-117 (FLEX data, Proc. Combust. Inst. 2023).")
    with t2:
        st.markdown("**A small gas flame**, like a leaking fuel or gas line. Will it **put itself out**? Based on **ACME CFI-G: "
                    "132 spherical gas flames on the ISS** (ambient O₂ 18–40%, 7–44 psi).")
        a, b = st.columns([1, 1.6])
        gf = a.radio("Gas", gm.fuels, horizontal=True, index=gm.fuels.index("propane") if "propane" in gm.fuels else 0)
        frac = a.select_slider("Fuel strength", [0.3, 0.5, 1.0], value=1.0, format_func=lambda v: f"{v:.0%} fuel (rest nitrogen)")
        go2 = a.slider("Oxygen (%)", 15, 45, int(min(max(o2 * 100, 15), 45)), key="g_o2") / 100
        gp = a.slider("Pressure (psi)", 6.0, 45.0, float(min(max(p_psi, 6.0), 45.0)), 0.5, key="g_p") / PSI
        gr = gm.predict(gf, frac, go2, gp)
        lvl, msg = reliability(gr["distance"])
        with b:
            c = st.columns(2)
            kpi(c[0], f"{gr['p_self_ext']:.0%}", "chance the flame goes out on its own", "green" if gr["p_self_ext"] > .5 else "red")
            kpi(c[1], f"{1 - gr['p_self_ext']:.0%}", "chance it keeps burning until the fuel is shut off", "orange")
            (st.success if lvl == "high" else st.warning if lvl == "medium" else st.error)(msg)
            xs = np.linspace(0.18, 0.42, 30)
            ys = [gm.predict(gf, frac, x, gp)["p_self_ext"] for x in xs]
            fig = go.Figure(go.Scatter(x=xs * 100, y=np.array(ys) * 100, line=dict(color="#0050D8", width=3), name=gf))
            fig.add_vline(x=21, line_dash="dot", annotation_text="Earth air")
            fig.add_vline(x=30, line_dash="dot", annotation_text="Moon base", line_color="#FF5804")
            fig.update_layout(title="Chance a gas flame puts itself out", xaxis_title="Oxygen (%)", yaxis=dict(title="%", range=[0, 105]))
            st.plotly_chart(light(fig, 340), width="stretch")
        st.caption("In CFI-G, every flame at 20–30% O₂ went out on its own, but a third kept burning at 30–40% O₂.")
    with t3:
        st.markdown("**How smoke changes before it reaches a detector**, from **SAME-R: 109 smoke tests on the ISS**. As smoke "
                    "ages, its particles grow, which changes what detectors see.")
        a, b = st.columns([1, 1.6])
        mat = a.selectbox("Material that's smouldering", sm.materials)
        v = a.slider("Air flow at the source (cm/s)", 0.0, 8.1, 8.1, 0.1)
        age = a.slider("Smoke travel / aging time (s)", 20, 1900, 60, 10)
        g_ = sm.predict(mat, v, age)
        with b:
            kpi(st.columns(1)[0], f"+{(g_ - 1) * 100:.0f}%", f"particle size growth for {mat} after {age} s", "orange")
            fig = go.Figure()
            for m_ in sm.materials:
                fig.add_bar(x=[m_], y=[(sm.predict(m_, v, age) - 1) * 100], marker_color="#FF5804" if m_ == mat else "#94A3B8")
            fig.update_layout(title=f"Particle growth by material ({age} s aging)", yaxis_title="% growth", showlegend=False)
            st.plotly_chart(light(fig, 320), width="stretch")
        st.caption("Detectors far from a fire see larger particles than detectors close by. Calibrate for the materials on board.")
    with t4:
        st.markdown("**One model across two experiments and two fuel types**: FLEX liquid droplets plus CFI-G gas flames "
                    "(405 flames). It answers the core question: **does more oxygen stop fires from going out on their own?**")
        cp = st.slider("Pressure (psi)", 7.0, 30.0, float(min(max(p_psi, 7.0), 30.0)), 0.5, key="pool_p") / PSI
        xs = np.linspace(0.15, 0.40, 40)
        fig = go.Figure()
        fig.add_scatter(x=xs * 100, y=pm.curve(xs, cp, 1) * 100, name="💧 liquid droplet (FLEX)", line=dict(color="#16A34A", width=3))
        fig.add_scatter(x=xs * 100, y=pm.curve(xs, cp, 0) * 100, name="💨 gas flame (CFI-G)", line=dict(color="#0050D8", width=3))
        fig.add_vline(x=21, line_dash="dot", annotation_text="Earth air")
        fig.add_vline(x=30, line_dash="dot", annotation_text="Moon base", line_color="#FF5804")
        fig.update_layout(title=f"Chance a flame goes out on its own, at {cp * PSI:.1f} psi", xaxis_title="Oxygen (%)",
                          yaxis=dict(title="%", range=[0, 105]), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(light(fig, 420), width="stretch")
        l21, l30 = pm.curve(np.array([0.21, 0.30]), cp, 1)
        g21, g30 = pm.curve(np.array([0.21, 0.30]), cp, 0)
        st.markdown(f"**Both experiments point the same way:** with more oxygen, flames are less likely to go out on their own. "
                    f"Liquid droplets: **{l21:.0%} → {l30:.0%}** from Earth air to Moon-base air. Gas flames: **{g21:.0%} → {g30:.0%}**. "
                    "The model is *physics-constrained*: more oxygen can never make self-extinction more likely, matching the raw data.")
    with t5:
        st.markdown("Every model is checked with **5-fold cross-validation**: each test is predicted by a model that never saw it.")
        mt = pd.DataFrame([
            ("💧 Liquid fuel: burning rate", "FLEX (PSI-69)", fm.metrics["burn_rate"]["n"], f"R² = {fm.metrics['burn_rate']['r2']:.2f}", "—"),
            ("💧 Liquid fuel: flame lifetime", "FLEX (PSI-69)", fm.metrics["burn_time"]["n"], f"R² = {fm.metrics['burn_time']['r2']:.2f}", "—"),
            ("💧 Liquid fuel: goes out on its own", "FLEX (PSI-69)", fm.metrics["self_extinguished"]["n"],
             f"{fm.metrics['self_extinguished']['accuracy']:.0%} correct", f"{max(fm.metrics['self_extinguished']['base_rate'], 1 - fm.metrics['self_extinguished']['base_rate']):.0%}"),
            ("💨 Gas flame: goes out on its own", "ACME CFI-G (PSI-159)", gm.metrics["n"], f"{gm.metrics['accuracy']:.0%} correct", f"{gm.metrics['base_rate']:.0%}"),
            ("🌫️ Smoke particle growth", "SAME-R (PSI-101)", sm.metrics["n"], f"R² = {sm.metrics['r2']:.2f}", "—"),
            ("🔗 All fuels: goes out on its own", "FLEX + CFI-G", pm.metrics["n"], f"{pm.metrics['accuracy']:.0%} correct", f"{pm.metrics['base_rate']:.0%}"),
        ], columns=["Model", "NASA data", "Tests", "Accuracy", "Guessing the most common outcome"])
        st.table(mt)
        st.markdown("**Tried and dropped** (not good enough to show):")
        for name, why in REJECTED:
            st.markdown(f"- {name}: {why}")
        st.caption("Methods: gradient-boosted trees (scikit-learn); quantile regression for likely ranges; oxygen-monotonic constraint "
                   "on the self-extinction models; extrapolation warning from distance to the nearest real test.")

# ================================================================== GAPS
elif ss.view == "gaps":
    loc, o2, p_psi = current()
    top_bar("🕳️ Where is NASA's data missing?")
    explain("Dots are real NASA tests. Empty areas are conditions nobody has tested in microgravity. The lower sections list "
            "the most useful tests to run next and let you check a specific material.")
    moon = known[(known.o2_fraction >= 0.30) & (known.pressure_atm * PSI <= 10.2)]
    c = st.columns(3)
    kpi(c[0], f"{len(known):,}", "tests with recorded oxygen and pressure", "blue")
    kpi(c[1], len(moon), "inside the planned Moon-base atmosphere", "orange")
    kpi(c[2], int(moon.fuel_phase.str.startswith("solid").sum()), "of them on solid cabin materials", "red")
    fig = build_figure(tests)
    fig.add_scatter(x=[o2 * 100], y=[p_psi], mode="markers", name="This location",
                    marker=dict(symbol="x", size=18, color="#FF5804", line=dict(width=3)))
    st.plotly_chart(light(fig, 520), width="stretch")
    st.markdown("### 🌍 Which gravity levels have fire data?")
    gcols = st.columns(4)
    for col, (gname, gval, txt, tone) in zip(gcols, [
            ("Microgravity", "0 g", "19 ISS & Cygnus flight investigations", "green"),
            ("Moon", "0.16 g", "No data yet. FM2 and LUCI are planned", "red"),
            ("Mars", "0.38 g", "No data", "red"),
            ("Earth", "1 g", "Ground tests & 1-g reference burns", "blue")]):
        kpi(col, gval, f"<b>{gname}</b><br>{txt}", tone)
    st.markdown("**Careful:** you can't assume Moon or Mars fires fall *between* space and Earth. NASA's FM2 team notes some "
                "materials may burn at **lower** oxygen in lunar gravity than on Earth, because a weak upward flow can feed a flame "
                "without cooling it. Partial gravity may be the worst case, which is why it's the top gap.")
    st.markdown("### 🧪 The tests NASA should run next")
    gaps = gap_table(tests)
    for r in gaps.head(3).itertuples():
        st.markdown(f'<div class="risk red">🎯 <b>Test {r.fuel_type.lower()} at {r.o2_pct}% O₂ and {r.p_psi} psi</b> '
                    f'<span class="small">({r.condition}) · data coverage {r.coverage:.0%} · closest existing test: {r.nearest}</span></div>',
                    unsafe_allow_html=True)
    st.markdown("**Already on the way:** " + " · ".join(f'<a href="{kb[k]["url"]}" target="_blank">{kb[k]["title"].split(":")[0]}</a>'
                                                       for k in ("F1", "F2", "F3") if k in kb), unsafe_allow_html=True)
    if "F2" in kb:
        st.caption(kb["F2"]["text"])
    st.markdown("### 🗺️ Coverage map by fuel type")
    grp = st.radio("Show coverage for", list(PHASE_GROUPS), horizontal=True)
    o2s, ps = np.linspace(0.15, 1.0, 70), np.geomspace(3, 20, 60)
    z = coverage_grid(tests, PHASE_GROUPS[grp], o2s, ps)
    hm = go.Figure(go.Heatmap(x=o2s * 100, y=ps, z=z, zmin=0, zmax=1, colorbar=dict(title="coverage"),
                              colorscale=[[0, "#FEECEC"], [0.5, "#FDBA74"], [1, "#16A34A"]],
                              hovertemplate="%{x:.0f}% O₂, %{y:.1f} psi<br>coverage %{z:.2f}<extra></extra>"))
    for name, (ho2, hp) in {"ISS": (0.21, 14.7), "Moon base / rover / Mars": (0.30, 10.2), "Spacesuit": (1.0, 4.3)}.items():
        hm.add_scatter(x=[ho2 * 100], y=[hp], mode="markers+text", text=[name], textposition="top center",
                       marker=dict(symbol="star", size=15, color="#1B2B85"), showlegend=False)
    hm.update_layout(title=f"How well {grp.lower()} are covered (green = tested nearby, pale = no data)",
                     xaxis_title="Oxygen (%)", yaxis=dict(title="Pressure (psi)", type="log"))
    st.plotly_chart(light(hm, 440), width="stretch")
    with st.expander("Full gap table"):
        st.dataframe(gaps, hide_index=True, width="stretch")
    st.markdown("### 🧱 Will my material burn here?")
    fam = st.selectbox("Cabin material", list(MATERIALS))
    mt = material_tests(tests, fam)
    status, msg = material_verdict(mt, o2, p_psi)
    st.markdown(f'<div class="verdict {"green" if status == "covered" else "red" if status == "gap" else "yellow"}">'
                f'<b>{fam} at {o2:.0%} O₂ / {p_psi:.1f} psi:</b> {msg}</div>', unsafe_allow_html=True)
    if len(mt):
        st.dataframe(mt.assign(o2_pct=(mt.o2_fraction * 100).round(1), p_psi=(mt.pressure_atm * PSI).round(1))
                     [["investigation", "psi_id", "fuel", "o2_pct", "p_psi", "outcome", "o2_source"]]
                     .rename(columns={"o2_pct": "O₂ (%)", "p_psi": "pressure (psi)", "o2_source": "O₂ value"}),
                     hide_index=True, width="stretch", height=230)
    if "F1" in kb:
        st.info("🌕 **Partial gravity:** PSI has no Moon or Mars gravity data yet. " + kb["F1"]["text"])

# ================================================================== EQUIPMENT
elif ss.view == "equipment":
    loc, o2, p_psi = current()
    top_bar("🧯 What protects the crew?")
    explain("Three tabs, one per line of defence: putting fires out, detecting smoke, and choosing materials. Each chart is "
            "real NASA data, followed by the takeaway in one sentence.")
    t1, t2, t3 = st.tabs(["🧯 Putting fires out", "🔔 Detecting fires", "🛡️ Choosing materials"])
    with t1:
        s_ = suppression_summary(fm.df)
        fig = go.Figure()
        for dname, color in [("Nitrogen (air-like)", "#94A3B8"), ("CO₂", "#FF5804"), ("Helium", "#0050D8")]:
            g_ = s_[s_.diluent == dname]
            fig.add_bar(x=g_.o2_band.astype(str), y=g_.self_ext_pct, name=dname, marker_color=color,
                        text=[f"{v}% (n={n})" for v, n in zip(g_.self_ext_pct, g_.tests)], textposition="outside")
        fig.update_layout(barmode="group", title="Share of FLEX droplet flames that went out on their own",
                          xaxis_title="Oxygen in the chamber", yaxis=dict(title="% went out on their own", range=[0, 115]))
        st.plotly_chart(light(fig, 400), width="stretch")
        st.markdown("**Takeaway:** below 15% O₂, over 90% of flames went out on their own; above 25% O₂, only 8% did. Lowering "
                    "oxygen is the strongest suppressant. *Source: FLEX (PSI-69).*")
        if len(bass_ar):
            oc = bass_ar.outcome.value_counts()
            fig = go.Figure(go.Bar(x=oc.values, y=oc.index, orientation="h", marker_color="#1B2B85"))
            fig.update_layout(title="How 106 BASS solid-material fires on the ISS ended (crew notes)", xaxis_title="tests")
            st.plotly_chart(light(fig, 320), width="stretch")
            st.caption("From the BASS 'As Run' test matrix in PSI (measured O₂ 16–23%, ~1 atm).")
        st.markdown("#### 🌬️ Will turning off the fans put the fire out?")
        bre = tests[tests.psi_id == "PSI-20"]
        n_off = int((bass_ar.outcome == "went out when airflow stopped").sum()) if len(bass_ar) else 0
        n_b2 = int(bass2.flow_to_extinction.sum()) if len(bass2) else 0
        v1, v2 = st.columns(2)
        v1.markdown(f'<div class="card"><b>❌ Not for gas-fed flames</b><br>ACME BRE lit 194 flames over {bre.test_id.nunique()} test days with <b>zero airflow</b>, '
                    f'at {bre.o2_fraction.min():.0%}–{bre.o2_fraction.max():.0%} O₂, including Moon-base/rover air (34%, 8.2 psi). '
                    'They kept burning steadily.<br><span class="small">Source: ACME BRE (PSI-20)</span></div>', unsafe_allow_html=True)
        v2.markdown(f'<div class="card"><b>✅ Sometimes for solid materials</b><br>In BASS, <b>{n_off} of {len(bass_ar)}</b> solid fires went out '
                    f'when the crew stopped the airflow, and BASS-II slowed the flow until flames died in <b>{n_b2} of {len(bass2)}</b> runs. But crew notes '
                    'also show a flame wake surviving a nitrogen jet with the air off.<br><span class="small">Source: BASS As Run (PSI-26), '
                    'BASS-II (PSI-25)</span></div>', unsafe_allow_html=True)
        st.markdown("**Takeaway:** fan shutdown helps some fires and not others, and the highest-oxygen data (BRE) says it fails. "
                    "Moon and Mars habitats need active suppression, not ventilation shutdown alone.")
        if "L4" in kb:
            st.markdown(f'<div class="lesson"><b>📜 {kb["L4"]["title"]}</b><br>{kb["L4"]["text"]}<br><span class="small">{kb["L4"]["source"]}</span></div>',
                        unsafe_allow_html=True)
    with t2:
        sr = pd.read_csv(raw_path("PSI-101__SAME-R.csv"), encoding="latin-1")
        g_ = sr.groupby("Sample Type")["Rt/R0"].median().sort_values()
        fig = go.Figure(go.Bar(x=g_.values, y=g_.index, orientation="h", marker_color="#D97706",
                               text=[f"+{(v - 1) * 100:.0f}%" for v in g_.values], textposition="outside"))
        fig.update_layout(title="How much smoke particles grow as smoke ages (SAME-R)", xaxis=dict(title="size after ÷ size before", range=[1, 1.6]))
        st.plotly_chart(light(fig, 320), width="stretch")
        st.markdown("**Takeaway:** smoke particles grew 23–45% as they aged, depending on the material. Detectors far from a "
                    "fire see larger particles. *Sources: SAME-R (PSI-101), SAME (PSI-102).*")
        for k in ("L7", "L5", "L6"):
            if k in kb:
                st.markdown(f'<div class="lesson"><b>📜 {kb[k]["title"]}</b><br>{kb[k]["text"]}<br><span class="small">{kb[k]["source"]}</span></div>',
                            unsafe_allow_html=True)
    with t3:
        st.markdown("**Earth flammability tests don't reliably predict space fires** (Saffire-II, PSI-99, burned on a Cygnus cargo ship):")
        st.table(SAFFIRE2)
        if len(bass2):
            n_ext = int(bass2.flow_to_extinction.sum())
            st.markdown(f"**BASS-II turned the airflow down until flames went out in {n_ext} of {len(bass2)} runs**, mapping how "
                        "little airflow a solid flame needs to survive.")
            with st.expander("See the BASS-II flow-limit runs"):
                st.dataframe(bass2, hide_index=True, width="stretch")
        for k in ("L1", "L2", "L3"):
            if k in kb:
                st.markdown(f'<div class="lesson"><b>📜 {kb[k]["title"]}</b><br>{kb[k]["text"]}<br><span class="small">{kb[k]["source"]}</span></div>',
                            unsafe_allow_html=True)

# ================================================================== LESSONS
elif ss.view == "lessons":
    top_bar("📜 What NASA has learned")
    explain("Real spacecraft fires, the key lessons from decades of ISS experiments, and the experiments coming next. "
            "Each card links to its NASA source.")
    sect = [("🚨 Past spacecraft fires", "history"), ("🔬 Key lessons from microgravity experiments", "lesson"),
            ("🚀 Coming next", "next")]
    for head, kind in sect:
        st.markdown(f"### {head}")
        items = [k for k in KB if k["kind"] == kind]
        cols = st.columns(3 if kind != "lesson" else 2)
        for i, k in enumerate(items):
            cols[i % len(cols)].markdown(f'<div class="lesson"><b>{k["title"]}</b><br>{k["text"]}<br>'
                                         f'<span class="small"><a href="{k["url"]}" target="_blank">{k["source"]}</a></span></div>',
                                         unsafe_allow_html=True)

# ================================================================== ATMOSPHERE TRADE-OFF
elif ss.view == "atmosphere":
    loc, o2, p_psi = current()
    top_bar("⚖️ Choose the best atmosphere")
    explain("Each point on this chart is a possible cabin atmosphere. Lower-left means a shorter wait before spacewalks; "
            "right of the orange line means fires burn more readily. Colour shows how much NASA data exists there.")
    st.markdown("Before a spacewalk, astronauts breathe pure oxygen to clear nitrogen from their blood. **Lower pressure with more "
                "oxygen shortens that wait**, which is why the Moon base is planned at ≥30% O₂ and ≤10.2 psi. But extra oxygen "
                "makes fires worse, and that's where NASA's fire data runs out.")
    grp = st.radio("Colour the chart by NASA data for", list(PHASE_GROUPS), horizontal=True)
    habs = {"ISS": (0.21, 14.7), "Moon base": (0.30, 10.2), "8.2 psi / 34%": (0.34, 8.2), "Spacesuit": (1.0, 4.3)}
    st.plotly_chart(light(tradeoff_figure(tests, coverage_grid, PHASE_GROUPS[grp], o2, p_psi, habs), 600), width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><b>Dotted lines</b><br><span class="small">Pre-breathe index: lower = shorter wait before a '
                'spacewalk. ISS 2.7, Moon base 1.66.</span></div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><b>Orange line</b><br><span class="small">Right of it is an "atmosphere of increased burning" '
                '(NFPA 99B).</span></div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><b>Green dashed line</b><br><span class="small">Same oxygen pressure as sea level, so it feels '
                'like normal air to breathe.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="verdict yellow">💡 <b>The insight:</b> below about 11.8 psi, every atmosphere that keeps normal breathing '
                'oxygen falls in the <b>increased-burning</b> zone, and that is where NASA has no microgravity data for solid '
                'cabin materials.</div>', unsafe_allow_html=True)
    rows = []
    for name, (ho2, hp) in habs.items():
        fl = atmosphere_flags(ho2, hp)
        rows.append({"Atmosphere": name, "O₂": f"{ho2:.0%}", "Pressure (psi)": hp, "Pre-breathe index": round(fl["prebreathe_index"], 2),
                     "Increased burning?": "yes" if fl["increased_burning"] else "no",
                     **{f"NASA data: {k.split(' (')[0]}": f"{coverage(tests, ho2, hp, v)[0]:.0%}" for k, v in PHASE_GROUPS.items()}})
    st.table(pd.DataFrame(rows))
    st.caption("NFPA 99B limit: O₂% > 23.45/√P (atm), as cited by NASA White Sands research. The pre-breathe index is a relative "
               "comparison, not a medical criterion.")

# ================================================================== DATA LIBRARY
elif ss.view == "data":
    top_bar("📚 Explore NASA's data")
    explain("Everything we used, in one place. Browse the 24 investigations, compare up to three, plot the tests, or download "
            "the clean dataset for your own analysis.")
    c = st.columns(3)
    kpi(c[0], len(inv), "combustion investigations in NASA PSI", "blue")
    kpi(c[1], f"{len(tests):,}", "tests cleaned into one shared format", "green")
    kpi(c[2], "16", "test tables merged (O₂, pressure, flow, fuel, outcome)", "orange")
    st.write("")
    d1, d2 = st.columns([1, 2])
    d1.download_button("⬇️ Download the full cleaned dataset (CSV)", tests.to_csv(index=False), "flame_in_freefall_tests.csv")
    d2.markdown('<span class="small">Columns: psi_id, investigation, test_id, fuel, fuel_phase, o2_fraction, pressure_atm, flow_cm_s, '
                'diluent, outcome, burn_time_s, o2_source, p_source (measured / listed / assumed_cabin_air). CC0. Cite each '
                'investigation by its DOI.</span>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["📖 Library", "⚖️ Compare", "📊 Plot the tests"])
    with t1:
        f1, f2 = st.columns(2)
        role = f1.selectbox("Fire-safety role", ["All", "prevention", "detection", "suppression", "fundamental"],
                            format_func=lambda x: "All" if x == "All" else PILLAR_ICON[x])
        phase = f2.selectbox("Fuel type", ["All", "solid", "liquid", "gas"], format_func=lambda x: "All" if x == "All" else PHASE_ICON[x])
        cols = st.columns(2)
        shown = 0
        for r in inv.sort_values("n_publications", ascending=False).itertuples():
            pillars, ph, link = CURATED.get(r.id, (["fundamental"], "gas", None))
            if role != "All" and role not in pillars:
                continue
            if phase != "All" and not ph.startswith(phase):
                continue
            t = tests[tests.psi_id == r.id]
            obj = str(r.objective).split(". ")[0]
            obj = obj if len(obj) < 170 else obj[:170].rsplit(" ", 1)[0] + "…"
            col = cols[shown % 2]
            col.markdown(f'<div class="card"><b>{name_of[r.id]}</b> <span class="small">({r.id} · {r.project_type.split()[0]})</span><br>'
                         + "".join(f'<span class="chip">{PILLAR_ICON.get(p, p)}</span>' for p in pillars)
                         + f'<span class="chip">{PHASE_ICON.get(ph, ph)}</span><span class="chip">{len(t)} tests</span>'
                         f'<span class="chip">{int(r.n_publications)} papers</span><br><span class="small">{obj}</span><br>'
                         f'<span class="small"><a href="{psi_url(r.id)}" target="_blank">PSI page</a>'
                         + (f' · <a href="https://doi.org/{r.doi}" target="_blank">DOI {r.doi}</a>' if isinstance(r.doi, str) else "")
                         + (f' · uses {name_of.get(link, link)} flight data' if link else "") + "</span></div>", unsafe_allow_html=True)
            if len(t):
                col.download_button(f"⬇️ {len(t)} tests (CSV)", t.to_csv(index=False), f"{r.id}_tests.csv", key=f"dl_{r.id}")
            shown += 1
        if not shown:
            st.info("No investigations match these filters.")
    with t2:
        ranked = score(inv, tests, *[current()[1], current()[2] / PSI])
        pick = st.multiselect("Pick 2–3 investigations", ranked.id.tolist(), default=ranked.id.head(3).tolist(),
                              format_func=lambda i: f"{name_of[i]} ({i})", max_selections=3)
        rows = {}
        for pid in pick:
            m, t = inv[inv.id == pid].iloc[0], tests[tests.psi_id == pid]
            pillars, ph, _ = CURATED.get(pid, (["fundamental"], "gas", None))
            rng = lambda s, k, u: f"{s.min() * k:.3g}–{s.max() * k:.3g}{u}" if len(t) and s.notna().any() else "not recorded"
            rows[name_of[pid]] = {"Role": ", ".join(PILLAR_ICON.get(p, p) for p in pillars), "Fuel type": PHASE_ICON.get(ph, ph),
                                  "Fuels": ", ".join(str(x) for x in t.fuel.value_counts().index[:4]) or "—",
                                  "O₂ tested": rng(t.o2_fraction, 100, " %"), "Pressure tested": rng(t.pressure_atm, PSI, " psi"),
                                  "Hardware": m.hardware if isinstance(m.hardware, str) else "—",
                                  "Tests · papers": f"{len(t)} · {m.n_publications}", "DOI": m.doi}
        if rows:
            st.table(pd.DataFrame(rows))
    with t3:
        sel = st.multiselect("Investigations", sorted(tests.investigation.unique()), default=["FLEX"])
        t = tests[tests.investigation.isin(sel)] if sel else tests
        labels = {"o2_fraction": "Oxygen (fraction)", "pressure_atm": "Pressure (atm)", "flow_cm_s": "Air flow (cm/s)",
                  "burn_time_s": "Burn time (s)"}
        a, b, c3 = st.columns(3)
        x = a.selectbox("X axis", list(labels), 0, format_func=labels.get)
        y = b.selectbox("Y axis", list(labels), 3, format_func=labels.get)
        color = c3.selectbox("Colour by", ["fuel", "diluent", "outcome", "investigation", "fuel_phase"])
        fig = go.Figure()
        for k, gg in t.dropna(subset=[x, y]).groupby(color):
            fig.add_scatter(x=gg[x], y=gg[y], mode="markers", name=str(k), marker=dict(size=8))
        fig.update_layout(xaxis_title=labels[x], yaxis_title=labels[y])
        st.plotly_chart(light(fig, 420), width="stretch")

# ================================================================== ASK
elif ss.view == "ask":
    loc, o2, p_psi = current()
    top_bar("💬 Ask a question")
    explain("Type a question or tap an example. Answers draw on the 24 PSI investigations, NASA's Researcher's Guide to ISS "
            "Combustion Science and NASA fire-history pages, and cite their sources.")
    st.caption("Answers come only from NASA PSI data and cite investigations. If the data doesn't cover it, it says so.")

    def docs():
        out = []
        for r in inv.fillna("").itertuples():
            t = tests[tests.psi_id == r.id]
            env = (f"Tests: {len(t)}; O2 {t.o2_fraction.min():.2f}-{t.o2_fraction.max():.2f}; pressure {t.pressure_atm.min():.2f}-"
                   f"{t.pressure_atm.max():.2f} atm; fuels: {', '.join(str(x) for x in t.fuel.dropna().unique()[:8])}"
                   if len(t) and t.o2_fraction.notna().any() else "No O2/pressure test table.")
            out.append(f"[{r.id} | {name_of[r.id]} | DOI {r.doi}]\n{r.title}\nObjective: {r.objective}\nImpacts: {r.research_impacts}\n"
                       f"{env}\nAI attributes: {json.dumps(enriched.get(r.id, {}))}")
        for k in KB:
            out.append(f"[{k['id']} | {k['title']} | {k['source']}]\nObjective: {k['text']}")
        for g_ in GUIDE:
            out.append(f"[{g_['id']} | Researcher's Guide to ISS Combustion Science | NASA 2015]\nObjective: {g_['text']}")
        return out

    def src_link(tag):
        if tag.startswith("PSI-"):
            return f"[{tag} · {name_of.get(tag, '')}]({psi_url(tag)})"
        if tag in kb:
            return f"[{kb[tag]['title']}]({kb[tag]['url']})"
        if tag.startswith("G") and GUIDE:
            return f"[NASA Researcher's Guide to ISS Combustion Science, passage {tag[1:]}]({GUIDE[0]['url']})"
        return tag

    def retrieve(q, k=6):
        from sklearn.feature_extraction.text import TfidfVectorizer
        D = docs()
        v = TfidfVectorizer(stop_words="english").fit(D + [q])
        sims = (v.transform(D) @ v.transform([q]).T).toarray().ravel()
        return [D[i] for i in sims.argsort()[::-1][:k]]

    examples = [f"Which experiments matter most for the {loc['name']}?", "What makes a flame go out on its own in microgravity?",
                "Has any solid material been tested above 30% oxygen?", "How does smoke behave in microgravity?"]
    ec = st.columns(2)
    for i, ex in enumerate(examples):
        ec[i % 2].button(ex, key=f"ex{i}", on_click=lambda e=ex: ss.update(ask=e), width="stretch")
    q = st.chat_input("Ask about fire safety in space…") or ss.pop("ask", None)
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
                st.warning("AI answers are off (no API key set). The most relevant investigations:")
                for c in ctx:
                    tag = c[1:].split(" |")[0]
                    st.markdown(f"**{src_link(tag)}**: {c.split('Objective: ')[1][:280]}…")
            else:
                import anthropic
                with st.spinner("Reading the investigations…"):
                    msg = anthropic.Anthropic(api_key=key).messages.create(
                        model=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"), max_tokens=900,
                        system=("You are a spacecraft fire-safety research assistant. Answer ONLY from the context, in plain English. "
                                "Cite every claim with its bracketed tag, e.g. [PSI-69], [L2] or [G14]. If the context does not cover "
                                "the question, say so. Under 200 words. "
                                f"Location: {loc['name']}, {o2:.0%} O2 at {p_psi:.1f} psi."),
                        messages=[{"role": "user", "content": "CONTEXT:\n\n" + "\n\n".join(ctx) + f"\n\nQUESTION: {q}"}])
                st.markdown(msg.content[0].text)
                st.caption("AI-generated answer (Claude, Anthropic), grounded in the sources below.")
            with st.expander("Sources used"):
                for c in ctx:
                    st.markdown("- " + src_link(c[1:].split(" |")[0]))

# ================================================================== EARTH
elif ss.view == "earth":
    st.button("← Back", on_click=go_view, args=("home",))
    st.markdown("## 🌍 Moon-base air already exists on Earth")
    st.markdown("Hospitals, hyperbaric medicine and industry all work in **oxygen-enriched air**, where the same question "
                "applies: how do everyday materials burn with extra oxygen?")
    st.plotly_chart(light(earth_figure(tests), 520), width="stretch")
    c1, c2 = st.columns(2)
    c1.markdown('<div class="card"><b>🏥 Why it matters</b><br>Hyperbaric chamber fires have killed patients: 11 people died in a '
                '1.8-atm chamber in Milan (1997), and a recent fatal fire in Troy, Michigan happened at 2 atm in ~100% O₂ with '
                'polyester and cotton linens inside.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><b>🔁 Space ↔ Earth</b><br>NASA\'s fabric data stops at ~22% O₂, the same gap facing a Moon base. '
                'New tests would help hospitals everywhere, including in Bahrain and the Gulf, choose safer materials.</div>',
                unsafe_allow_html=True)
    st.table(EARTH.rename(columns={"setting": "Setting", "note": "Note"})[["Setting", "Note"]])
    st.caption("Sources: NFPA 99 / OSHA (>23.5% O₂ = oxygen-enriched); hyperbaric practice (WHA International, Baromedical); "
               "NBDHMT 2026 fire report. Links on the About page.")

# ================================================================== ABOUT
elif ss.view == "about":
    st.button("← Back", on_click=go_view, args=("home",))
    st.markdown("## ℹ️ About Flame in Freefall")
    st.markdown(f"""
**Challenge:** NASA Space Apps 2026, *Flame in Freefall: SI-Powered Fire Safety Insights from Microgravity Combustion Data*.

**Data:** all {len(inv)} combustion investigations in NASA Physical Sciences Informatics (PSI), harvested on 14 Nov 2026 (CC0).
{len(tests):,} tests from 16 experimental tables, converted to one format. Raw videos (≈11 TB) were not used.

**Assumptions:** where a table doesn't record O₂ or pressure (SPICE, SLICE, SAME, Saffire, a few BASS tests), ISS cabin air is
assumed and flagged. BASS now uses measured values from its As Run matrix. Moon base, rover and spacesuit atmospheres come from the challenge statement; Gateway and the Mars transit ship are
assumed ISS-like; the Mars base is assumed equal to the Moon base. PSI has no partial-gravity data.

**Methods:** relevance score (fire-safety focus, closeness of tests, fuel relevance, data richness); coverage = distance to the
nearest real test; four cross-validated machine-learning models (see *Predict a fire → How accurate?*).
""")
    st.markdown("""
**Sources beyond PSI**
- Oxygen-enriched atmosphere (NFPA 99, OSHA): [WHA International](https://wha-international.com/what-is-an-oxygen-enriched-atmosphere/)
- NFPA 99B increased burning (O₂% > 23.45/√P): [Beeson, NASA White Sands, ASTM](https://store.astm.org/stp12040s.html)
- Hyperbaric fires: [WHA](https://wha-international.com/?p=1962) · [NBDHMT 2026 report](https://nbdhmt.org/wp-content/uploads/2026/03/Catastrophic-Hyperbaric-Chamber-Fire-Report-dated-March-27-2026.pdf) · [Baromedical](https://baromedical.com/_files/ugd/e43c3a_e36f07ddab924991884056bf2a67bf24.pdf)
- [NASA Researcher's Guide to ISS Combustion Science (2015)](https://www.nasa.gov/missions/station/a-researchers-guide-to-combustion-science/) ·
  [NASA Glenn Combustion Science](https://www.nasa.gov/glenn/glenn-expertise-space-exploration/physical-sciences-program/combustion-science/) ·
  [Apollo 1](https://sma.nasa.gov/sma-disciplines/accllp/apollo-1) · [Apollo 13](https://sma.nasa.gov/sma-disciplines/accllp/apollo-13) ·
  [Mir fire](https://www.nasa.gov/history/25-years-ago-fire-aboard-space-station-mir/)
- NASA spinoffs: [fire detector](https://spinoff.nasa.gov/node/9407) · [air tanks](https://spinoff.nasa.gov/node/9582) · [fire attire](https://spinoff.nasa.gov/node/9918)

**Use of AI (Super Intelligence)**
- Parts of the code were written with the help of Claude (Anthropic) and reviewed by the team.
- Chat answers (and optional AI summaries) are generated by Claude and labelled in the app.
- Reports, fire-risk cards and verdicts are rule-based from the data, not AI-generated.
- No AI-generated images. The Space Apps logo and BASS photo are official files, unmodified.
""")
    st.markdown("""
**More data harvested from PSI and the challenge resources**
- BASS *As Run* test matrix: measured O₂/pressure and crew-noted outcomes for 106 solid-material tests (replaces assumed air)
- BASS-II test matrices: 129 runs, 31 with airflow turned down until the flame went out
- PSI-117 analysed FLEX data: hot vs cool flame extinction diameters; FLEX raw droplet-diameter histories
- NASA *Researcher's Guide to ISS Combustion Science* (2015): 86 passages searchable in the chat, plus key lessons
- NASA Glenn Combustion Science pages (FM2, SoFIE, FLARE) and NASA fire-history pages (Apollo 1, Apollo 13, Mir)

**Glossary**

| Term | Meaning |
|---|---|
| Microgravity (µg) | Near-weightlessness, as on the ISS. Hot gases don't rise, so flames are round. |
| Partial gravity | Moon (1/6 g) or Mars (3/8 g). Not yet in PSI; FM2 will be the first lunar fire test. |
| Diluent | The non-oxygen gas in the air (nitrogen on Earth). CO₂ and helium were tested as suppressants. |
| Burning rate | How fast a fuel droplet shrinks as it burns (mm²/s). |
| Self-extinguished | The flame went out on its own before the fuel was used up. |
| Cool flame | A low-temperature, nearly invisible flame that can keep reacting after the visible flame dies. |
| NFPA 99B "increased burning" | Air with more than 23.45/√P(atm) % oxygen, where fire codes require extra protection. |
| PSI ID / DOI | NASA's dataset identifier and the permanent citation link for each investigation. |

**Benefits on Earth:** fine-water-mist extinguishers developed for the ISS · a shuttle-era smoke detector that tells smoke
from dust (now used in mining) · firefighter air tanks and protective clothing based on spacesuit technology.
""")

_want = {"view": ss.view, **({"loc": ss.loc} if ss.loc else {})}
if dict(st.query_params) != _want:
    st.query_params.from_dict(_want)

st.markdown(f'<div class="small" style="margin-top:30px;border-top:1px solid #DFE1E2;padding-top:10px">'
            f'<img src="{LOGO}" style="height:26px;vertical-align:middle;margin-right:10px;background:#1B2B85;padding:3px 6px;border-radius:6px">'
            'Flame in Freefall · NASA Space Apps 2026 · Data: NASA PSI (CC0) · Built with AI assistance (see About)</div>',
            unsafe_allow_html=True)
