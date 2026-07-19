"""Dashboard charts for the Fable (2027) public-data analysis.

Run: pip install plotly pandas kaleido && python charts.py
Writes four HTML files (interactive, with hover tooltips) next to this script.

Palette: validated CVD-safe reference palette (light mode).
Rules applied: single hue per single-series chart, no legend for one series,
direct labels, diverging blue-gray-red only for polarity (sentiment),
ordinal blue steps for the feature matrix, no pie charts, one axis per chart.
"""

import plotly.graph_objects as go

# --- palette (light mode, validated) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"      # categorical slot 1 / sequential hue
RED = "#e34948"       # diverging warm pole
NEUTRAL = "#898781"   # diverging neutral gray (validated: >=3:1 contrast, CVD-safe vs blue/red)
BASELINE = "#c3c2b7"  # axis/baseline gray

LAYOUT = dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif",
              color=INK, size=13),
    margin=dict(l=10, r=30, t=60, b=40),
)


def timeline_chart() -> go.Figure:
    """1. Marketing & release timeline — single series, direct labels."""
    events = [
        ("2020-07-23", "Anunț (Xbox Showcase)", False),
        ("2023-06-11", "Primul trailer in-game", False),
        ("2024-06-09", "Fereastră 2025 anunțată", False),
        ("2025-02-25", "Amânarea #1 → 2026", True),
        ("2026-01-22", "Developer Direct: gameplay", False),
        ("2026-05-29", "Amânarea #2 → feb 2027", True),
        ("2026-06-08", "Precomenzi deschise", False),
        ("2027-02-23", "Lansare planificată", False),
    ]
    fig = go.Figure()
    for date, label, is_delay in events:
        fig.add_trace(go.Scatter(
            x=[date], y=[0], mode="markers+text",
            marker=dict(size=12, color=RED if is_delay else BLUE,
                        line=dict(width=2, color=SURFACE)),
            text=[label], textposition="top center",
            textfont=dict(size=11, color=INK_2),
            hovertemplate=f"{label}<br>{date}<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        title="Fable (2027) — cronologie marketing (roșu = amânare)",
        yaxis=dict(visible=False, range=[-1, 1.5]),
        xaxis=dict(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED)),
        **LAYOUT,
    )
    return fig


def feature_matrix() -> go.Figure:
    """2. Feature matrix vs Fable 1-3 — ordinal heatmap + direct text labels."""
    features = ["Open world real", "1.000+ NPC-uri vocate", "Combat fluid melee/ranged/magic",
                "Moralitate", "Romance & familie", "Proprietăți & afaceri",
                "Co-op", "Lansare PlayStation"]
    games = ["Fable 1 (2004)", "Fable 2 (2008)", "Fable 3 (2010)", "Fable (2027)"]
    # 0 = absent, 1 = partial, 2 = complet
    z = [
        [0, 0, 0, 2],
        [0, 0, 0, 2],
        [1, 1, 1, 2],
        [2, 2, 2, 2],
        [1, 2, 2, 2],
        [1, 2, 2, 2],
        [0, 2, 2, 0],
        [0, 0, 0, 2],
    ]
    labels = {0: "✗", 1: "~", 2: "✓"}
    # ordinal blue ramp: steps 250 / 450 / 650, plus near-surface for absent
    colorscale = [[0.0, "#f0efec"], [0.33, "#f0efec"],
                  [0.34, "#86b6ef"], [0.66, "#86b6ef"],
                  [0.67, "#2a78d6"], [1.0, "#2a78d6"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=games, y=features, colorscale=colorscale, showscale=False,
        xgap=2, ygap=2,  # 2px surface gap between cells
        text=[[labels[v] for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=14),
        hovertemplate="%{y} — %{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Feature matrix: Fable (2027) vs. trilogia clasică (✓ complet, ~ parțial, ✗ absent)",
        yaxis=dict(autorange="reversed", tickfont=dict(color=INK_2)),
        xaxis=dict(tickfont=dict(color=INK_2)),
        **LAYOUT,
    )
    return fig


def sentiment_chart() -> go.Figure:
    """3. Community sentiment — 100% stacked bar (NOT a pie).

    Values are a qualitative estimate from forum/press coverage, not a
    measured dataset — the title says so. Diverging encoding: polarity.
    """
    segments = [("Pozitiv", 45, BLUE), ("Mixt / în așteptare", 35, NEUTRAL),
                ("Negativ / sceptic", 20, RED)]
    fig = go.Figure()
    for name, val, color in segments:
        fig.add_trace(go.Bar(
            y=["Sentiment"], x=[val], name=name, orientation="h",
            marker=dict(color=color, line=dict(width=2, color=SURFACE)),
            text=[f"{name} ~{val}%"], textposition="inside",
            insidetextfont=dict(color=SURFACE),
            hovertemplate=f"{name}: ~{val}% (estimare)<extra></extra>",
        ))
    fig.update_layout(
        title="Sentiment comunitate (estimare calitativă, iul 2026 — nu date măsurate)",
        barmode="stack", legend=dict(orientation="h", y=-0.3),
        xaxis=dict(range=[0, 100], ticksuffix="%", gridcolor=GRID,
                   tickfont=dict(color=MUTED)),
        yaxis=dict(visible=False), height=260,
        **LAYOUT,
    )
    return fig


def price_chart() -> go.Figure:
    """4. Edition price comparison — single hue, direct labels, no legend."""
    editions = ["Standard", "Premium (digital)", "Collector's"]
    prices = [69.99, 99.99, 199.99]
    fig = go.Figure(go.Bar(
        y=editions[::-1], x=prices[::-1], orientation="h",
        marker=dict(color=BLUE, cornerradius=4,
                    line=dict(width=2, color=SURFACE)),
        text=[f"${p:.2f}" for p in prices[::-1]], textposition="outside",
        textfont=dict(color=INK),
        hovertemplate="%{y}: $%{x:.2f}<extra></extra>", width=0.55,
    ))
    fig.update_layout(
        title="Fable (2027) — prețuri ediții (USD)",
        xaxis=dict(gridcolor=GRID, linecolor=BASELINE, tickprefix="$",
                   tickfont=dict(color=MUTED), range=[0, 230]),
        yaxis=dict(tickfont=dict(color=INK_2)), height=320, showlegend=False,
        **LAYOUT,
    )
    return fig


# 5. Platform availability: deliberately a table, not a bar chart —
# availability is binary, a bar would imply a magnitude that doesn't exist.
PLATFORM_MATRIX = """
| Platformă        | Standard | Premium (digital) | Collector's | Game Pass |
|------------------|----------|-------------------|-------------|-----------|
| Xbox Series X|S  | ✓        | ✓                 | ✓           | ✓ day one |
| PC (Windows)     | ✓        | ✓                 | ✓           | ✓ day one |
| PlayStation 5    | ✓        | ✓                 | ✓           | —         |
"""

if __name__ == "__main__":
    for name, fig in [("timeline", timeline_chart()),
                      ("feature_matrix", feature_matrix()),
                      ("sentiment", sentiment_chart()),
                      ("prices", price_chart())]:
        fig.write_html(f"fable_{name}.html", include_plotlyjs="cdn")
        print(f"wrote fable_{name}.html")
    print(PLATFORM_MATRIX)
