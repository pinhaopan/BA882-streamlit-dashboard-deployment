import os
from typing import List, Dict

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans

from google.cloud import secretmanager

MOTHERDUCK_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InBwMDEyNkBidS5lZHUiLCJtZFJlZ2lvbiI6ImF3cy11cy1lYXN0LTEiLCJzZXNzaW9uIjoicHAwMTI2LmJ1LmVkdSIsInBhdCI6IklZd01ST2w2LWU5RFRITTBnMHRjdXk2MG9aNVJKOGhkREM2LUE1UEl3Sk0iLCJ1c2VySWQiOiI4OWFmZmFkYS0yMjMzLTQ1YTgtOWE5NS03NTdhMTJhZDNjNjciLCJpc3MiOiJtZF9wYXQiLCJyZWFkT25seSI6ZmFsc2UsInRva2VuVHlwZSI6InJlYWRfd3JpdGUiLCJpYXQiOjE3NjMzMjU3OTF9.WnPryE-58CngLwKWpu0zZisU2OZStz4BiTaSRYHXuSY" 
os.environ["MOTHERDUCK_TOKEN"] = MOTHERDUCK_TOKEN

def get_secret(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")  


@st.cache_resource
def get_connection():
    """
    Connect to the `ncaa` MotherDuck database using the environment variable MOTHERDUCK_TOKEN.
    """
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        st.error(
            "Environment variable `MOTHERDUCK_TOKEN` is not set.\n\n"
            "Please run in your terminal / Cloud Shell:\n"
            "    export MOTHERDUCK_TOKEN='md:your_token_here'\n\n"
            "Then restart Streamlit."
        )
        st.stop()

    md_db_name = "ncaa"
    conn = duckdb.connect(f"md:{md_db_name}")
    return conn


def run_query(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Helper: execute SQL and return a DataFrame."""
    conn = get_connection()
    if params is None:
        return conn.execute(sql).fetch_df()
    return conn.execute(sql, params).fetch_df()


# ------------------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------------------
st.set_page_config(page_title="BA882 College Football Dashboard", layout="wide")
st.title("🏈 NCAA 2025 College Football Analytics Dashboard")
st.caption("Data source: ESPN Hidden API → GCS → MotherDuck (via Airflow)")

SEASON = 2025 

# ============================================================================
# 2025 Season Team Stats + Benchmark (Median)
# ============================================================================
TEAM_METRICS = [
    # Offense
    "avg_points_scored",
    "avg_total_yards",
    "avg_yards_per_pass",
    "avg_yards_per_rush",
    "points_per_yard_offense",
    # Defense
    "avg_points_allowed",
    "avg_yards_allowed",
    "avg_third_eff_allowed",
    "avg_fourth_eff_allowed",
    "points_per_yard_defense",
    # Overall
    "win_pct",
    "point_differential",
    "turnover_margin",
]


# Metrics where higher is better
HIGHER_IS_BETTER = {
    "avg_points_scored": True,
    "avg_total_yards": True,
    "avg_yards_per_pass": True,
    "avg_yards_per_rush": True,
    "points_per_yard_offense": True,
    "avg_points_allowed": False,     
    "avg_yards_allowed": False,
    "avg_third_eff_allowed": False,
    "avg_fourth_eff_allowed": False,
    "points_per_yard_defense": False,
    "win_pct": True,
    "point_differential": True,
    "turnover_margin": True,
}


@st.cache_data(show_spinner="Loading team stats & benchmark (median)...")
def load_team_stats_with_benchmark() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load team metrics from bt.team_stats + dim_teams,
    and compute the league median benchmark for all metrics.
    """
    # Detect whether team_stats contains a season column
    probe = run_query("SELECT * FROM bt.team_stats LIMIT 1;")
    has_season = "season" in probe.columns

    base_sql = """
        SELECT
            ts.*,
            COALESCE(t.display_name, t.name) AS team_name
        FROM bt.team_stats AS ts
        LEFT JOIN real_deal.dim_teams AS t
            ON ts.team_id = t.id
    """
    if has_season:
        base_sql += " WHERE ts.season = ?"

    df = run_query(base_sql, (SEASON,) if has_season else None)

    # Keep rows with valid team names
    df = df.dropna(subset=["team_name"])

    # Compute league median benchmark
    bench = df[TEAM_METRICS].median(numeric_only=True)

    return df, bench


@st.cache_data(show_spinner="Loading 2025 games metadata...")
def load_games_meta() -> pd.DataFrame:
    sql = """
        SELECT
            id,
            season,
            week,
            start_date,
            venue_id
        FROM real_deal.dim_games
        WHERE season = ?
    """
    return run_query(sql, (SEASON,))


# ============================================================================
# Utility Functions – Percentile / Colors / Normalization
# ============================================================================
def compute_percentiles(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    Compute 0–100 percentile scores for each metric.
    Higher/better or lower/better rules follow HIGHER_IS_BETTER.
    """
    result = df.copy()
    for col in metric_cols:
        if col not in result.columns:
            continue
        values = result[col].astype(float)
        order = values.rank(method="average", pct=True)  # 0–1
        if not HIGHER_IS_BETTER.get(col, True):
            order = 1 - order
        result[col + "_pctile"] = (order * 100).round(1)
    return result


def normalize_for_radar(team_row: pd.Series, benchmark: pd.Series, cols: List[str]) -> Dict[str, List[float]]:
    """
    Convert team vs benchmark values into radar chart–friendly normalized ratios.
    For offense (higher is better): value / benchmark.
    For defense (lower is better): benchmark / value.
    """
    team_vals = []
    bench_vals = []
    for col in cols:
        t = float(team_row[col])
        b = float(benchmark[col])
        if b == 0:
            team_vals.append(1.0)
            bench_vals.append(1.0)
            continue

        if HIGHER_IS_BETTER.get(col, True):
            team_vals.append(t / b)
            bench_vals.append(1.0)
        else:
            team_vals.append(b / t if t != 0 else 1.0)
            bench_vals.append(1.0)

    return {"team": team_vals, "bench": bench_vals}


def metric_label(col: str) -> str:
    """Beautify metric names for display."""
    mapping = {
        "avg_points_scored": "Avg Points Scored",
        "avg_total_yards": "Avg Total Yards",
        "avg_yards_per_pass": "Yards per Pass",
        "avg_yards_per_rush": "Yards per Rush",
        "points_per_yard_offense": "Points per Yard (Offense)",
        "avg_points_allowed": "Avg Points Allowed",
        "avg_yards_allowed": "Avg Yards Allowed",
        "avg_third_eff_allowed": "Opp 3rd Down Eff",
        "avg_fourth_eff_allowed": "Opp 4th Down Eff",
        "points_per_yard_defense": "Points per Yard (Defense)",
        "win_pct": "Win %",
        "point_differential": "Point Differential",
        "turnover_margin": "Turnover Margin",
    }
    return mapping.get(col, col)


# ============================================================================
# Sidebar – Main Navigation
# ============================================================================
page = st.sidebar.radio(
    "📌 Dashboard Sections:",
    [
        "1. Overview",
        "2. Team Performance",
        "3. Ranking Evolution",
        "4. Head-to-Head",
        "5. League Analytics",
    ],
)

# ============================================================================
# 1. Overview
# ============================================================================
if page == "1. Overview":
    st.subheader("📊 2025 Season Overview")

    with st.spinner("Loading league overview..."):
        df_stats, benchmark = load_team_stats_with_benchmark()
        df_games = load_games_meta()

    # 1-1) Overall KPIs
    col1, col2, col3 = st.columns(3)
    n_teams = df_stats["team_id"].nunique()
    n_games = df_games["id"].nunique()
    avg_win_pct = df_stats["win_pct"].mean()

    col1.metric("Teams in 2025", n_teams)
    col2.metric("Games in 2025", n_games)
    col3.metric("Average Win %", f"{avg_win_pct:.3f}")

    st.markdown("---")

    # 1-2) Metric Distribution (Histogram / Box)
    st.markdown("### 📈 League KPI Distribution")

    metric_for_dist = st.selectbox(
        "Choose a metric to display:",
        TEAM_METRICS,
        format_func=metric_label,
    )

    dist_col1, dist_col2 = st.columns(2)

    # Histogram
    hist_fig = px.histogram(
        df_stats,
        x=metric_for_dist,
        nbins=20,
        title=f"{metric_label(metric_for_dist)} – Distribution",
    )
    hist_fig.add_vline(
        x=benchmark[metric_for_dist],
        line_dash="dash",
        annotation_text="League Median",
        annotation_position="top left",
    )
    dist_col1.plotly_chart(hist_fig, use_container_width=True)

    # Box Plot
    if "conference" in df_stats.columns:
        box_fig = px.box(
            df_stats,
            x="conference",
            y=metric_for_dist,
            title=f"{metric_label(metric_for_dist)} by Conference",
        )
    else:
        box_fig = px.box(
            df_stats,
            y=metric_for_dist,
            title=f"{metric_label(metric_for_dist)} – Box Plot",
        )
    dist_col2.plotly_chart(box_fig, use_container_width=True)

    st.markdown("---")

    # 1-3) Top Performers Table
    st.markdown("### 🏅 Top 10 Teams by Metric")

    metric_for_top = st.selectbox(
        "Choose ranking metric:",
        TEAM_METRICS,
        index=TEAM_METRICS.index("win_pct"),
        format_func=metric_label,
        key="overview_top_metric",
    )

    df_pct = compute_percentiles(df_stats, [metric_for_top])
    m_col = metric_for_top
    higher_is_better = HIGHER_IS_BETTER.get(metric_for_top, True)

    df_top = df_pct[["team_name", m_col, m_col + "_pctile"]].dropna()
    df_top = df_top.sort_values(by=m_col, ascending=not higher_is_better).head(10)

    st.dataframe(
        df_top.rename(
            columns={
                "team_name": "Team",
                m_col: metric_label(m_col),
                m_col + "_pctile": "Percentile (0-100)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # 1-4) League Heatmap – team vs median
    st.markdown("### 🔥 League Performance Heatmap (vs Median)")

    heat_df = df_stats[["team_name"] + TEAM_METRICS].copy()
    for col in TEAM_METRICS:
        if HIGHER_IS_BETTER.get(col, True):
            heat_df[col] = heat_df[col] / benchmark[col]
        else:
            heat_df[col] = benchmark[col] / heat_df[col]

    heat_matrix = heat_df.set_index("team_name")[TEAM_METRICS]

    heat_fig = px.imshow(
        heat_matrix,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        origin="lower",
        labels=dict(color="Relative to Median"),
        title="Team Performance vs League Median (>1 = above median)",
    )
    st.plotly_chart(heat_fig, use_container_width=True)


# ============================================================================
# 2. Team Performance
# ============================================================================
elif page == "2. Team Performance":
    st.subheader("🎯 Team Performance vs League Benchmark")

    with st.spinner("Loading team stats & benchmark..."):
        df_stats, benchmark = load_team_stats_with_benchmark()
        df_pct = compute_percentiles(df_stats, TEAM_METRICS)

    team_options = df_stats["team_name"].sort_values().unique().tolist()
    selected_team = st.selectbox("Select a team", team_options)

    team_row = df_pct[df_pct["team_name"] == selected_team].iloc[0]

    # 2-1) KPI Cards
    st.markdown("### 📌 Key KPIs vs League Median")

    kpi_cols = ["win_pct", "avg_points_scored", "avg_points_allowed", "point_differential"]
    kpi_display = [metric_label(c) for c in kpi_cols]

    cols = st.columns(len(kpi_cols))
    for c, label, col_container in zip(kpi_cols, kpi_display, cols):
        val = float(team_row[c])
        med = float(benchmark[c])
        delta_pct = (val - med) / med if med != 0 else 0.0
        col_container.metric(
            label=label,
            value=f"{val:.3f}" if c == "win_pct" else f"{val:.1f}",
            delta=f"{delta_pct:+.1%}",
        )

    st.markdown("---")

    # 2-2) Radar Chart
    st.markdown("### 🕸 Radar – Multi-dimensional Comparison vs Benchmark")

    radar_metrics = [
        "win_pct",
        "avg_points_scored",
        "avg_points_allowed",
        "avg_total_yards",
        "avg_yards_allowed",
        "turnover_margin",
    ]
    labels = [metric_label(c) for c in radar_metrics]
    radar_data = normalize_for_radar(team_row, benchmark, radar_metrics)

    radar_fig = go.Figure()
    radar_fig.add_trace(
        go.Scatterpolar(
            r=radar_data["team"],
            theta=labels,
            fill="toself",
            name=selected_team,
        )
    )
    radar_fig.add_trace(
        go.Scatterpolar(
            r=radar_data["bench"],
            theta=labels,
            fill="toself",
            name="League Median",
        )
    )
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.5, max(max(radar_data["team"]), 1.5)])),
        showlegend=True,
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("---")

    # 2-3) Bar chart – detailed comparison
    st.markdown("### 📊 Detailed Metric Comparison (Team vs Median)")

    metric_multi = st.multiselect(
        "Select metrics to compare:",
        TEAM_METRICS,
        default=["avg_points_scored", "avg_points_allowed", "win_pct", "point_differential"],
        format_func=metric_label,
    )

    long_rows = []
    for m in metric_multi:
        long_rows.append({"metric": metric_label(m), "who": selected_team, "value": float(team_row[m])})
        long_rows.append({"metric": metric_label(m), "who": "League Median", "value": float(benchmark[m])})
    comp_df = pd.DataFrame(long_rows)

    bar_fig = px.bar(
        comp_df,
        x="metric",
        y="value",
        color="who",
        barmode="group",
        title="Team vs League Median",
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("---")

    # 2-4) Percentile table
    st.markdown("### 🪜 Percentile Ranking by Metric")

    table_rows = []
    for m in TEAM_METRICS:
        table_rows.append(
            {
                "Metric": metric_label(m),
                "Value": float(team_row[m]),
                "Percentile (0-100, higher = better)": float(team_row[m + "_pctile"]),
            }
        )
    pct_df = pd.DataFrame(table_rows)
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2-5) Performance Gauges
    st.markdown("### 🧭 Performance Gauges")

    off_pct = float(team_row["points_per_yard_offense_pctile"])
    def_pct = float(team_row["points_per_yard_defense_pctile"])
    overall_pct = float(team_row["win_pct_pctile"])

    g1, g2, g3 = st.columns(3)

    def gauge(col, value, title):
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": title},
                gauge={"axis": {"range": [0, 100]}},
            )
        )
        col.plotly_chart(fig, use_container_width=True)

    gauge(g1, off_pct, "Offense Efficiency (percentile)")
    gauge(g2, def_pct, "Defense Efficiency (percentile)")
    gauge(g3, overall_pct, "Overall Performance (Win% percentile)")


# ============================================================================
# 3. Ranking Evolution
# ============================================================================
elif page == "3. Ranking Evolution":
    st.subheader("📉 Bradley–Terry Ranking – Evolution & Analytics")

    with st.spinner("Loading rankings & team stats..."):
        df_hist = run_query(
            """
            SELECT
                h.team_id,
                COALESCE(t.display_name, t.name) AS team_name,
                h.rank,
                h.strength,
                h.prob_vs_avg,
                h.updated_at
            FROM bt.model_ranking_history AS h
            LEFT JOIN real_deal.dim_teams AS t
                ON h.team_id = t.id
            ORDER BY h.updated_at
            """
        )
        df_stats, benchmark = load_team_stats_with_benchmark()

    if df_hist.empty:
        st.warning("The table bt.model_ranking_history is currently empty.")
    else:
        team_options = df_hist["team_name"].dropna().unique().tolist()
        selected_team = st.selectbox("Select a team to view ranking history", sorted(team_options))

        df_team = df_hist[df_hist["team_name"] == selected_team].sort_values("updated_at")

        c1, c2 = st.columns(2)

        rank_fig = px.line(
            df_team,
            x="updated_at",
            y="rank",
            title=f"{selected_team} – Rank over time (lower is better)",
        )
        rank_fig.update_yaxes(autorange="reversed")
        c1.plotly_chart(rank_fig, use_container_width=True)

        strength_fig = px.line(
            df_team,
            x="updated_at",
            y="strength",
            title=f"{selected_team} – Strength over time",
        )
        c2.plotly_chart(strength_fig, use_container_width=True)

        st.markdown("---")

    st.markdown("### 🔗 Current Ranking vs Performance Metrics")

    df_rank_now = run_query(
        """
        SELECT
            r.team_id,
            COALESCE(t.display_name, t.name) AS team_name,
            r.rank,
            r.strength,
            r.prob_vs_avg
        FROM bt.rankings AS r
        LEFT JOIN real_deal.dim_teams AS t
            ON r.team_id = t.id
        ORDER BY r.rank
        """
    )

    df_join = pd.merge(
        df_rank_now,
        df_stats[["team_id"] + TEAM_METRICS],
        on="team_id",
        how="left",
    )

    metric_for_corr = st.selectbox(
        "Choose a metric for correlation with Rank:",
        TEAM_METRICS,
        index=TEAM_METRICS.index("win_pct"),
        format_func=metric_label,
        key="rank_metric",
    )

    scat_col1, scat_col2 = st.columns(2)

    fig_scatter = px.scatter(
        df_join,
        x=metric_for_corr,
        y="rank",
        hover_name="team_name",
        trendline="ols",
        title=f"Rank vs {metric_label(metric_for_corr)}",
    )
    fig_scatter.update_yaxes(autorange="reversed")
    scat_col1.plotly_chart(fig_scatter, use_container_width=True)

    corr_val = df_join[[metric_for_corr, "rank"]].corr().iloc[0, 1]
    scat_col2.metric(
        "Correlation (Rank vs Metric, Pearson)",
        f"{corr_val:.3f}",
        help="Negative = higher metric correlates with better rank (smaller number).",
    )

    st.markdown("#### Current ranking table")
    st.dataframe(df_rank_now, use_container_width=True, hide_index=True)


# ============================================================================
# 4. Head-to-Head Comparison
# ============================================================================
elif page == "4. Head-to-Head":
    st.subheader("⚔️ Head-to-Head Team Comparison (vs Median)")

    with st.spinner("Loading team stats & benchmark..."):
        df_stats, benchmark = load_team_stats_with_benchmark()
        df_pct = compute_percentiles(df_stats, TEAM_METRICS)

    team_options = df_stats["team_name"].sort_values().unique().tolist()
    selected_teams = st.multiselect(
        "Choose 2–4 teams to compare:",
        team_options,
        default=team_options[:2],
    )

    if len(selected_teams) < 2:
        st.info("Please select at least 2 teams.")
        st.stop()
    if len(selected_teams) > 4:
        st.warning("Maximum recommended comparison is 4 teams; using the first 4.")
        selected_teams = selected_teams[:4]

    df_sel = df_pct[df_pct["team_name"].isin(selected_teams)].copy()

    st.markdown("### 📋 Summary Table – Metrics & Percentiles")

    rows = []
    for _, row in df_sel.iterrows():
        for m in TEAM_METRICS:
            rows.append(
                {
                    "Team": row["team_name"],
                    "Metric": metric_label(m),
                    "Value": float(row[m]),
                    "Percentile (0-100)": float(row[m + "_pctile"]),
                }
            )
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True)

    st.markdown("---")

    st.markdown("### 🕸 Radar – Multi-team Comparison vs Median")

    radar_metrics = [
        "win_pct",
        "avg_points_scored",
        "avg_points_allowed",
        "avg_total_yards",
        "avg_yards_allowed",
        "turnover_margin",
    ]
    radar_labels = [metric_label(c) for c in radar_metrics]

    radar_fig = go.Figure()

    bench_data = normalize_for_radar(benchmark, benchmark, radar_metrics)
    radar_fig.add_trace(
        go.Scatterpolar(
            r=bench_data["bench"],
            theta=radar_labels,
            fill="toself",
            name="League Median",
            opacity=0.4,
        )
    )

    for _, row in df_sel.iterrows():
        data = normalize_for_radar(row, benchmark, radar_metrics)
        radar_fig.add_trace(
            go.Scatterpolar(
                r=data["team"],
                theta=radar_labels,
                fill="toself",
                name=row["team_name"],
                opacity=0.6,
            )
        )

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.5, 2.0])),
        showlegend=True,
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### 📊 Relative to Median – Strength / Weakness Matrix")

    metric_for_bar = st.selectbox(
        "Choose a metric:",
        TEAM_METRICS,
        index=TEAM_METRICS.index("avg_points_scored"),
        format_func=metric_label,
    )

    bar_rows = []
    for _, row in df_sel.iterrows():
        val = float(row[metric_for_bar])
        med = float(benchmark[metric_for_bar])
        diff_pct = (val - med) / med if med != 0 else 0
        bar_rows.append(
            {
                "Team": row["team_name"],
                "Value": val,
                "Diff vs Median (%)": diff_pct * 100,
            }
        )
    bar_df = pd.DataFrame(bar_rows)

    col_val, col_diff = st.columns(2)

    fig_val = px.bar(
        bar_df,
        x="Team",
        y="Value",
        title=f"{metric_label(metric_for_bar)} – Absolute Values",
    )
    col_val.plotly_chart(fig_val, use_container_width=True)

    fig_diff = px.bar(
        bar_df,
        x="Team",
        y="Diff vs Median (%)",
        title=f"{metric_label(metric_for_bar)} – Difference vs Median (%)",
    )
    col_diff.plotly_chart(fig_diff, use_container_width=True)


# ============================================================================
# 5. League Analytics
# ============================================================================
elif page == "5. League Analytics":
    st.subheader("📚 League-wide Analytics – Distribution, Correlation, Clusters")

    with st.spinner("Loading league stats..."):
        df_stats, benchmark = load_team_stats_with_benchmark()

    tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation", "Clustering"])

    # -------------------------------
    # Tab 1 – Distribution
    # -------------------------------
    with tab1:
        st.markdown("### 📈 Metric Distribution")

        metric_for_dist = st.selectbox(
            "Choose a metric:",
            TEAM_METRICS,
            index=TEAM_METRICS.index("win_pct"),
            format_func=metric_label,
            key="league_dist_metric",
        )

        col1, col2 = st.columns(2)

        fig_hist = px.histogram(
            df_stats,
            x=metric_for_dist,
            nbins=20,
            title=f"{metric_label(metric_for_dist)} – Histogram",
        )
        fig_hist.add_vline(
            x=benchmark[metric_for_dist],
            line_dash="dash",
            annotation_text="Median",
            annotation_position="top left",
        )
        col1.plotly_chart(fig_hist, use_container_width=True)

        fig_box = px.box(
            df_stats,
            y=metric_for_dist,
            title=f"{metric_label(metric_for_dist)} – Box Plot",
        )
        col2.plotly_chart(fig_box, use_container_width=True)

    # -------------------------------
    # Tab 2 – Correlation
    # -------------------------------
    with tab2:
        st.markdown("### 🔗 Correlation Heatmap (Team-level metrics)")

        corr_df = df_stats[TEAM_METRICS].corr()

        fig_corr = px.imshow(
            corr_df,
            x=[metric_label(c) for c in TEAM_METRICS],
            y=[metric_label(c) for c in TEAM_METRICS],
            color_continuous_scale="RdBu",
            origin="lower",
            title="Correlation between Metrics",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("### Scatter – Explore relationships")

        x_metric = st.selectbox(
            "X-axis metric:",
            TEAM_METRICS,
            index=TEAM_METRICS.index("avg_points_scored"),
            format_func=metric_label,
        )
        y_metric = st.selectbox(
            "Y-axis metric:",
            TEAM_METRICS,
            index=TEAM_METRICS.index("avg_points_allowed"),
            format_func=metric_label,
        )

        fig_sc = px.scatter(
            df_stats,
            x=x_metric,
            y=y_metric,
            hover_name="team_name",
            title=f"{metric_label(x_metric)} vs {metric_label(y_metric)}",
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # -------------------------------
    # Tab 3 – Clustering
    # -------------------------------
    with tab3:
        st.markdown("### 🧬 Clustering – Group teams by style / performance")

        cluster_metrics = st.multiselect(
            "Choose 2–8 metrics for clustering:",
            TEAM_METRICS,
            default=[
                "win_pct",
                "avg_points_scored",
                "avg_points_allowed",
                "avg_total_yards",
                "avg_yards_allowed",
            ],
            format_func=metric_label,
        )

        if len(cluster_metrics) < 2:
            st.info("Please select at least 2 metrics for clustering.")
        else:
            k = st.slider("Number of clusters (k)", min_value=2, max_value=6, value=3)

            X = df_stats[cluster_metrics].fillna(df_stats[cluster_metrics].median())
            X_std = (X - X.mean()) / X.std(ddof=0)

            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(X_std)

            df_cluster = df_stats.copy()
            df_cluster["cluster"] = labels

            st.markdown("#### Cluster assignment table")
            st.dataframe(
                df_cluster[["team_name", "cluster"] + cluster_metrics],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### 2D projection for visualization")

            x_metric = cluster_metrics[0]
            y_metric = cluster_metrics[1]

            fig_cluster = px.scatter(
                df_cluster,
                x=x_metric,
                y=y_metric,
                color="cluster",
                hover_name="team_name",
                title=f"Clusters by {metric_label(x_metric)} & {metric_label(y_metric)}",
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
