"""
Flame in Freefall - coverage map: every PSI test on O2 vs pressure, with habitat atmospheres.
Usage:  python coverage_map.py psi_combustion_data   -> coverage_map.html
In Streamlit: from coverage_map import build_figure; st.plotly_chart(build_figure(df))
"""
import math, sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

PSI_PER_ATM = 14.696
HABITATS = [  # (name, O2 %, pressure psi) - from the challenge statement
    ("Earth / ISS cabin", 21, 14.7),
    ("Moon base & rover (planned)", 30, 10.2),
    ("Spacesuit", 100, 4.3),
]
PHASE_COLORS = {"solid": "#d1495b", "solid (smoke)": "#e07b00", "liquid": "#2a9d4b", "gas": "#30638e"}
ORDER = ["gas", "liquid", "solid (smoke)", "solid"]  # solids drawn last so they stay visible


def build_figure(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["o2_fraction", "pressure_atm"]).copy()
    d["o2_pct"] = d["o2_fraction"] * 100
    d["p_psi"] = d["pressure_atm"] * PSI_PER_ATM
    d["assumed"] = (d["o2_source"] == "assumed_cabin_air") | (d["p_source"] == "assumed_cabin_air")
    moon = d[(d.o2_pct >= 30) & (d.p_psi <= 10.2)]

    fig = go.Figure()
    # Moon-base envelope: >=30% O2 at <=10.2 psi
    fig.add_shape(type="rect", x0=30, x1=100, y0=3, y1=10.2, fillcolor="rgba(237,174,73,0.12)",
                  line=dict(color="rgba(237,174,73,0.8)", dash="dot"), layer="below")
    fig.add_annotation(x=65, y=math.log10(8.8), text=f"Moon-base / rover envelope: {len(moon)} of {len(d)} tests",
                       showarrow=False, font=dict(size=12, color="#9a6a10"))

    for phase in sorted(d.fuel_phase.unique(), key=lambda p: ORDER.index(p) if p in ORDER else 0):
        g = d[d.fuel_phase == phase]
        for assumed, gg in g.groupby("assumed"):
            fig.add_trace(go.Scatter(
                x=gg.o2_pct, y=gg.p_psi, mode="markers",
                name=f"{phase}{' (cabin air assumed)' if assumed else ''}",
                marker=dict(size=11 if phase.startswith("solid") else 7, color=PHASE_COLORS.get(phase, "#888"), opacity=0.35 if assumed else 0.75,
                            symbol="circle-open" if assumed else "circle", line=dict(width=1)),
                customdata=gg[["investigation", "psi_id", "fuel", "test_id", "outcome"]].astype(str).values,
                hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>fuel: %{customdata[2]}<br>"
                              "test: %{customdata[3]}<br>outcome: %{customdata[4]}<br>"
                              "O2 %{x:.1f}% · %{y:.1f} psi<extra></extra>"))

    for name, o2, p in HABITATS:
        fig.add_trace(go.Scatter(x=[o2], y=[p], mode="markers+text", text=[name], textposition="middle left" if o2 > 90 else "top center",
                                 marker=dict(symbol="star", size=18, color="black"), showlegend=False,
                                 hovertemplate=f"<b>{name}</b><br>{o2}% O2 · {p} psi<extra></extra>"))

    fig.update_layout(
        title="Where NASA has burned things in microgravity vs. where crews will live",
        xaxis=dict(title="Oxygen concentration (% by volume)", range=[5, 102]),
        yaxis=dict(title="Pressure (psi, log scale)", type="log", range=[0.48, 1.9],
                   tickvals=[4, 5, 7, 10, 14.7, 20, 30, 45, 70], ticktext=["4", "5", "7", "10", "14.7", "20", "30", "45", "70"]),
        template="plotly_white", height=600, legend=dict(orientation="h", y=-0.18))
    return fig


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "psi_combustion_data")
    df = pd.read_csv(root / "tests_normalized.csv")
    fig = build_figure(df)
    fig.write_html(root / "coverage_map.html", include_plotlyjs="cdn")
    print("wrote", root / "coverage_map.html")
