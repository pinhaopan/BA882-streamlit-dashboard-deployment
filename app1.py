import os
from typing import List, Dict, Optional

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import re
from sklearn.cluster import KMeans
import requests

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


def run_query(sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """Helper: execute SQL and return a DataFrame."""
    conn = get_connection()
    if params is None:
        return conn.execute(sql).fetch_df()
    # DuckDB uses $1, $2, etc. for parameter placeholders, not ?
    # We need to replace ? with $1, $2, etc.
    param_count = sql.count('?')
    for i in range(param_count):
        sql = sql.replace('?', f'${i+1}', 1)
    return conn.execute(sql, params).fetch_df()


@st.cache_data(show_spinner=False)
def get_db_schema_text() -> str:
    """
    Query MotherDuck/DuckDB for schema info and return a compact text description
    for the LLM prompt. Only includes schemas bt and real_deal.
    """
    try:
        df = run_query(
            """
            SELECT
                table_schema,
                table_name,
                column_name
            FROM information_schema.columns
            WHERE table_schema IN ('bt', 'real_deal')
            ORDER BY table_schema, table_name, ordinal_position
            """
        )
    except Exception:
        return ""

    if df.empty:
        return ""

    lines = ["Available tables and columns:"]
    for (schema, table), group in df.groupby(["table_schema", "table_name"]):
        cols = ", ".join(group["column_name"].tolist())
        lines.append(f"{schema}.{table}({cols})")
    return "\n".join(lines)


def is_sql_safe(sql: str) -> bool:
    """Block obvious DDL/DML from running."""
    forbidden = re.compile(r"\b(insert|update|delete|alter|drop|create|truncate)\b", re.IGNORECASE)
    return not bool(forbidden.search(sql))


def _extract_sql_from_gemini(data: dict) -> str | None:
    """
    Gemini generateContent response → SQL text.
    Expected path: candidates[0].content.parts[0].text
    """
    try:
        candidates = data.get("candidates") or []
        part = candidates[0]["content"]["parts"][0]["text"]
        sql = part.strip()
        # Clean fenced code if present
        if sql.startswith("```"):
            sql = sql.strip("`")
            if "\n" in sql:
                sql = sql.split("\n", 1)[1]
        return sql.strip()
    except Exception:
        return None


def generate_sql_from_text(question: str) -> str | None:
    """
    Send the NL question to an LLM endpoint (Gemini-compatible) and return generated SQL.
    Requires `LLM_ENDPOINT` and `LLM_API_KEY` in st.secrets.
    """
    endpoint = st.secrets.get("LLM_ENDPOINT")
    api_key = st.secrets.get("LLM_API_KEY")
    if not endpoint or not api_key:
        st.warning("Missing LLM endpoint or API key in st.secrets.")
        return None

    schema_text = get_db_schema_text()

    # Prompt keeps the model constrained to SQL only.
    prompt = (
        "You are an assistant that writes DuckDB SQL for a MotherDuck database.\n"
        "Use ONLY the tables and columns listed below.\n"
        "Do NOT invent table or column names.\n"
        "Do NOT use placeholders like 'your_table_name'.\n"
        "When returning teams, join real_deal.dim_teams on team_id = id and display dim_teams.display_name as team_name.\n"
        "Assume the user provides full official team names; normalize to title case (so casing doesn't matter) and match exactly (case-insensitive) to dim_teams.display_name or dim_teams.name. Do NOT use wildcards that can match multiple teams.\n"
        "Use the correct tables for metrics: bt.team_stats (team_id, win_pct, avg_points_scored, avg_points_allowed, etc.), join via team_id.\n"
        "When filtering by season, use the actual column from the target table: fact_rankings uses season_year; bt.team_stats does NOT have a season column, so do not apply season filters to bt.team_stats unless you join to a table that has season_year. Do not invent column names.\n"
        "For poll_name, default to filtering within ('AFCA Coaches Poll','AP Poll','CFP Rankings'); only use other poll_name values if the user explicitly asks for bt rankings.\n"
        "If the user mentions BT/Bradley–Terry rankings, use bt.rankings; if they ask for BT ranking history, use bt.model_ranking_history. Apply rank filters/order accordingly.\n"
        "When querying rankings (non-BT), filter to the latest ingest_timestamp (max) for the chosen poll, so each rank maps to a single team (e.g., top 25 rows for AP Poll). For BT tables, do not assume a season_year column—use what exists in bt.rankings / bt.model_ranking_history (e.g., updated_at) and compute week_number if needed using the 2025-08-23 start rule.\n"
        "When using CTEs and UNION/UNION ALL, declare all CTEs at the top, then perform the UNION; do not place WITH clauses after a UNION.\n"
        "If comparing BT rankings vs another poll (e.g., CFP, AP, AFCA), produce a wide result with columns team_name, bt_rank, other_rank (one row per team). Use bt.rankings for BT ranks; for the other poll, take the latest ingest_timestamp from real_deal.fact_rankings with the requested poll_name. Do NOT output a long table with ranking_source.\n"
        "If the user asks for a ranking trend/history, return week-level rows (season_year, week_number, rank) ordered by week_number; only collapse to the latest ingest_timestamp when they want the current snapshot.\n"
        "For college football week_number in 2025, treat 2025-08-23 as the start of week 1 (not calendar ISO weeks); compute week_number as datediff('week', DATE '2025-08-23', the_date) + 1.\n"
        "For percentage-like fields (e.g., win_pct), multiply by 100 before displaying.\n"
        "Round numeric outputs to 2 decimal places (use ROUND(value, 2)).\n"
        "Return columns that are chart-friendly: include a categorical column (e.g., team_name or a label like 'Team' vs 'League Average') and separate numeric metric columns. For comparisons, output multiple rows or a label column instead of a single wide row.\n"
        "Return ONLY the SQL query, no explanation, no markdown fences.\n\n"
        f"{schema_text}\n\n"
        f"Question: {question}\n"
        "Write a single DuckDB SQL query that answers this question."
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
    }
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(
            endpoint,
            params={"key": api_key},
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        st.error(f"LLM request failed: {e}")
        return None

    # Try Gemini shape first; fall back to {sql: "..."} if a proxy returns that.
    sql = _extract_sql_from_gemini(data) or data.get("sql")
    if not sql:
        st.error("LLM response missing SQL content.")
        return None
    return sql.strip()


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
        WHERE season = 2025
    """
    return run_query(sql, (SEASON,))


# ============================================================================
# NEW: Season Overview Data Functions
# ============================================================================
@st.cache_data(show_spinner="Loading comprehensive season overview...")
def load_season_overview() -> dict:
    """Load comprehensive season overview metrics from pairwise_comparisons and dim_games"""
    
    try:
        # 1. From pairwise_comparisons - get game-level statistics
        games_sql = """
            SELECT 
                COUNT(*) AS total_games,
                AVG(home_score + away_score) AS avg_total_points,
                AVG(ABS(home_score - away_score)) AS avg_margin,
                MAX(home_score + away_score) AS highest_total,
                MIN(home_score + away_score) AS lowest_total,
                MAX(ABS(home_score - away_score)) AS biggest_margin,
                MIN(ABS(home_score - away_score)) AS closest_margin,
                SUM(CASE WHEN home_won = 1 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS home_win_pct
            FROM bt.pairwise_comparisons
        """
        games_stats = run_query(games_sql).iloc[0]  
    except Exception as e:
        st.error(f"Error loading from bt.pairwise_comparisons: {e}")
        st.info("Falling back to basic game statistics from dim_games...")
        # Fallback: use basic stats from dim_games
        games_sql_fallback = f"""
            SELECT 
                COUNT(*) AS total_games,
                50.0 AS avg_total_points,
                15.0 AS avg_margin,
                100 AS highest_total,
                10 AS lowest_total,
                50 AS biggest_margin,
                0 AS closest_margin,
                0.5 AS home_win_pct
            FROM real_deal.dim_games
            WHERE season = {SEASON}
        """
        games_stats = run_query(games_sql_fallback).iloc[0]
    
    # 2. From dim_games - get time ranges and weeks
    time_sql = f"""
        SELECT 
            MIN(start_date) AS season_start,
            MAX(start_date) AS latest_game,
            MIN(week) AS first_week,
            MAX(week) AS latest_week,
            COUNT(DISTINCT week) AS weeks_played
        FROM real_deal.dim_games
        WHERE season = {SEASON}
    """
    time_stats = run_query(time_sql).iloc[0]  
    
    # 3. From team_stats - get total teams and update timestamp
    team_sql = """
        SELECT 
            COUNT(DISTINCT team_id) AS total_teams,
            MAX(updated_at) AS data_updated_at
        FROM bt.team_stats
    """
    team_stats = run_query(team_sql).iloc[0]
    
    return {
        'total_games': int(games_stats['total_games']),
        'total_teams': int(team_stats['total_teams']),
        'avg_total_points': float(games_stats['avg_total_points']),
        'avg_margin': float(games_stats['avg_margin']),
        'home_win_pct': float(games_stats['home_win_pct']),
        'season_start': time_stats['season_start'],
        'latest_game': time_stats['latest_game'],
        'first_week': int(time_stats['first_week']),
        'latest_week': int(time_stats['latest_week']),
        'weeks_played': int(time_stats['weeks_played']),
        'data_updated_at': team_stats['data_updated_at'],
        'highest_total': int(games_stats['highest_total']),
        'lowest_total': int(games_stats['lowest_total']),
        'biggest_margin': int(games_stats['biggest_margin']),
        'closest_margin': int(games_stats['closest_margin']),
    }


@st.cache_data(show_spinner="Loading highlight games...")
def load_highlight_games():
    """Load notable games for highlights section"""
    try:
        sql = f"""
            SELECT 
                pc.game_id,
                pc.home_score,
                pc.away_score,
                pc.score_margin,
                pc.home_total_yards,
                pc.away_total_yards,
                ht.display_name AS home_team,
                away.display_name AS away_team,
                g.start_date,
                g.week
            FROM bt.pairwise_comparisons AS pc
            JOIN real_deal.dim_games AS g ON pc.game_id = g.id
            JOIN real_deal.dim_teams AS ht ON pc.home_team_id = ht.id
            JOIN real_deal.dim_teams AS away ON pc.away_team_id = away.id
            WHERE g.season = {SEASON}
            ORDER BY g.start_date DESC
        """
        df = run_query(sql)  
        df['total_points'] = df['home_score'] + df['away_score']
        df['point_diff'] = df['score_margin'].abs()
        return df
    except Exception as e:
        st.warning(f"Could not load highlight games from pairwise_comparisons: {e}")
        # Return empty dataframe with correct structure
        return pd.DataFrame(columns=[
            'game_id', 'home_score', 'away_score', 'score_margin', 
            'home_total_yards', 'away_total_yards', 'home_team', 'away_team',
            'start_date', 'week', 'total_points', 'point_diff'
        ])


@st.cache_data(show_spinner="Loading weekly trends...")
def load_weekly_trends():
    """Load game statistics aggregated by week"""
    try:
        sql = f"""
            SELECT 
                g.week,
                COUNT(*) AS games_played,
                AVG(pc.home_score + pc.away_score) AS avg_total_points,
                AVG(pc.home_score) AS avg_home_score,
                AVG(pc.away_score) AS avg_away_score,
                AVG(ABS(pc.home_score - pc.away_score)) AS avg_margin,
                SUM(CASE WHEN pc.home_won = 1 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS home_win_rate,
                SUM(CASE WHEN ABS(pc.home_score - pc.away_score) <= 7 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS close_game_rate
            FROM bt.pairwise_comparisons AS pc
            JOIN real_deal.dim_games AS g ON pc.game_id = g.id
            WHERE g.season = {SEASON}
            GROUP BY g.week
            ORDER BY g.week
        """
        return run_query(sql) 
    except Exception as e:
        st.warning(f"Could not load weekly trends from pairwise_comparisons: {e}")
        # Fallback: basic week counts from dim_games
        sql_fallback = f"""
            SELECT 
                week,
                COUNT(*) AS games_played,
                50.0 AS avg_total_points,
                28.0 AS avg_home_score,
                22.0 AS avg_away_score,
                12.0 AS avg_margin,
                0.55 AS home_win_rate,
                0.35 AS close_game_rate
            FROM real_deal.dim_games
            WHERE season = {SEASON}
            GROUP BY week
            ORDER BY week
        """
        return run_query(sql_fallback) 


@st.cache_data(show_spinner="Loading game distributions...")
def load_game_distributions():
    """Load individual game statistics for distribution analysis"""
    try:
        sql = f"""
            SELECT 
                pc.home_score,
                pc.away_score,
                pc.home_score + pc.away_score AS total_points,
                ABS(pc.home_score - pc.away_score) AS point_margin,
                pc.home_total_yards,
                pc.away_total_yards,
                pc.home_turnovers,
                pc.away_turnovers,
                g.week
            FROM bt.pairwise_comparisons AS pc
            JOIN real_deal.dim_games AS g ON pc.game_id = g.id
            WHERE g.season = {SEASON}
        """
        return run_query(sql)
    except Exception as e:
        st.warning(f"Could not load game distributions from pairwise_comparisons: {e}")
        # Return empty dataframe with correct structure
        return pd.DataFrame(columns=[
            'home_score', 'away_score', 'total_points', 'point_margin',
            'home_total_yards', 'away_total_yards', 'home_turnovers', 
            'away_turnovers', 'week'
        ])


@st.cache_data(show_spinner="Calculating interesting statistics...")
def calculate_interesting_stats():
    """Calculate interesting season-wide statistics"""
    sql = """
        SELECT 
            COUNT(*) AS total_games,
            SUM(CASE WHEN ABS(home_score - away_score) <= 7 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS close_game_pct,
            SUM(CASE WHEN ABS(home_score - away_score) <= 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS one_score_pct,
            SUM(CASE WHEN home_won = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS home_advantage,
            AVG(home_score + away_score) AS avg_combined,
            SUM(CASE WHEN home_score = 0 OR away_score = 0 THEN 1 ELSE 0 END) AS shutouts
        FROM bt.pairwise_comparisons
    """
    return run_query(sql).iloc[0]


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
        #"5. League Analytics",
        "6. Text to SQL",
    ],
)

# ============================================================================
# 1. Overview
# ============================================================================
if page == "1. Overview":
    st.subheader("📊 2025 Season Overview")
    
    # Section description
    st.markdown("""
    Get a comprehensive view of the 2025 NCAA Football season with key statistics, 
    trends over time, and notable performances.
    """)
    
    with st.spinner("Loading season data..."):
        # Load all necessary data
        overview_stats = load_season_overview()
        df_highlights = load_highlight_games()
        df_weekly = load_weekly_trends()
        df_game_dist = load_game_distributions()
        interesting_stats = calculate_interesting_stats()
        df_stats, benchmark = load_team_stats_with_benchmark()
    
    # Additional data cleaning for highlights
    # Force convert to numeric and drop any rows with NaN
    df_highlights['total_points'] = pd.to_numeric(df_highlights['total_points'], errors='coerce')
    df_highlights['point_diff'] = pd.to_numeric(df_highlights['point_diff'], errors='coerce')
    df_highlights = df_highlights.dropna(subset=['total_points', 'point_diff'])
    
    # Data freshness indicator
    st.caption(f"📊 Data as of: {overview_stats['data_updated_at'].strftime('%B %d, %Y')}")
    st.caption(f"📅 Season: {overview_stats['season_start'].strftime('%b %d')} → {overview_stats['latest_game'].strftime('%b %d, %Y')} | Weeks {overview_stats['first_week']}-{overview_stats['latest_week']}")
    
    st.markdown("---")
    
    # ========================================================================
    # 1-1) Season KPIs
    # ========================================================================
    st.markdown("### 📈 Season at a Glance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Teams", overview_stats['total_teams'])
    col2.metric("Games Played", overview_stats['total_games'])
    col3.metric("Weeks Played", f"{overview_stats['first_week']}-{overview_stats['latest_week']}")
    col4.metric("Avg Total Points", f"{overview_stats['avg_total_points']:.1f}")
    col5.metric("Home Win %", f"{overview_stats['home_win_pct']:.1%}")
    
    st.markdown("---")
    
    # ========================================================================
    # 1-2) Season Highlights
    # ========================================================================
    st.markdown("### 🔥 Season Highlights")
    
    # Find notable games
    highest_scoring = df_highlights.nlargest(1, 'total_points').iloc[0]
    lowest_scoring = df_highlights.nsmallest(1, 'total_points').iloc[0]
    biggest_blowout = df_highlights.nlargest(1, 'point_diff').iloc[0]
    closest_game = df_highlights.nsmallest(1, 'point_diff').iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 🔥 Highest Scoring")
        st.metric(
            label="Total Points",
            value=f"{int(highest_scoring['total_points'])} pts"
        )
        st.caption(f"{highest_scoring['home_team']} vs {highest_scoring['away_team']}")
        st.caption(f"{int(highest_scoring['home_score'])}-{int(highest_scoring['away_score'])} | Week {int(highest_scoring['week'])}")

    with col2:
        st.markdown("#### 🛡️ Defensive Battle")
        st.metric(
            label="Total Points",
            value=f"{int(lowest_scoring['total_points'])} pts"
        )
        st.caption(f"{lowest_scoring['home_team']} vs {lowest_scoring['away_team']}")
        st.caption(f"{int(lowest_scoring['home_score'])}-{int(lowest_scoring['away_score'])} | Week {int(lowest_scoring['week'])}")

    with col3:
        st.markdown("#### ⚡ Biggest Blowout")
        st.metric(
            label="Point Margin",
            value=f"{int(biggest_blowout['point_diff'])} pts"
        )
        st.caption(f"{biggest_blowout['home_team']} vs {biggest_blowout['away_team']}")
        st.caption(f"{int(biggest_blowout['home_score'])}-{int(biggest_blowout['away_score'])} | Week {int(biggest_blowout['week'])}")

    with col4:
        st.markdown("#### 🎯 Nail-Biter")
        st.metric(
            label="Point Margin",
            value=f"{int(closest_game['point_diff'])} pts"
        )
        st.caption(f"{closest_game['home_team']} vs {closest_game['away_team']}")
        st.caption(f"{int(closest_game['home_score'])}-{int(closest_game['away_score'])} | Week {int(closest_game['week'])}")
    
    st.markdown("---")
    
    # ========================================================================
    # 1-3) Did You Know?
    # ========================================================================
    st.markdown("### 💡 Did You Know?")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Close Games (≤7 pts)",
        f"{interesting_stats['close_game_pct']:.1%}",
        help="Percentage of games decided by one touchdown or less"
    )
    
    col2.metric(
        "One-Score Games (≤3 pts)",
        f"{interesting_stats['one_score_pct']:.1%}",
        help="Percentage of games decided by a field goal or less"
    )
    
    col3.metric(
        "Home Field Advantage",
        f"{interesting_stats['home_advantage']:.1%}",
        delta=f"{(interesting_stats['home_advantage'] - 0.5) * 100:+.1f}% vs 50%",
        help="Home team win percentage (50% would indicate no advantage)"
    )
    
    col4.metric(
        "Shutouts",
        f"{int(interesting_stats['shutouts'])}",
        help="Number of games where one team scored 0 points"
    )
    
    st.markdown("---")
    
    # ========================================================================
    # 1-4) Weekly Trends
    # ========================================================================
    st.markdown("### 📈 Season Trends Over Time")
    
    tab1, tab2, tab3 = st.tabs(["📊 Games & Scoring", "🏠 Home Advantage", "⚔️ Competitiveness"])
    
    with tab1:
        # Dual-axis chart: games played + avg scoring
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart for games played
        fig.add_trace(
            go.Bar(
                x=df_weekly['week'],
                y=df_weekly['games_played'],
                name="Games Played",
                marker_color='lightblue',
                opacity=0.6
            ),
            secondary_y=False
        )
        
        # Line chart for avg total points
        fig.add_trace(
            go.Scatter(
                x=df_weekly['week'],
                y=df_weekly['avg_total_points'],
                name="Avg Total Points",
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Week")
        fig.update_yaxes(title_text="Number of Games", secondary_y=False)
        fig.update_yaxes(title_text="Average Total Points", secondary_y=True)
        fig.update_layout(
            title="Games Played and Scoring by Week",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        busiest_week = df_weekly.loc[df_weekly['games_played'].idxmax()]
        st.info(f"🔥 **Busiest week:** Week {int(busiest_week['week'])} with {int(busiest_week['games_played'])} games")
        
        # Scoring trend
        early_weeks = df_weekly[df_weekly['week'] <= 4]
        late_weeks = df_weekly[df_weekly['week'] >= df_weekly['week'].max() - 3]
        
        if len(early_weeks) > 0 and len(late_weeks) > 0:
            early_avg = early_weeks['avg_total_points'].mean()
            late_avg = late_weeks['avg_total_points'].mean()
            change = late_avg - early_avg
            
            if abs(change) > 2:
                direction = "increased" if change > 0 else "decreased"
                st.info(f"📊 **Scoring trend:** Points have {direction} by {abs(change):.1f} from early season ({early_avg:.1f} pts) to late season ({late_avg:.1f} pts)")
                with st.expander("ℹ️ About this analysis", expanded=False):
                    st.markdown(f"""
                    **How we calculate seasonal trends:**
                    - **Early Season**: Weeks 1-4 (Average: {early_avg:.1f} pts/game)
                    - **Late Season**: Weeks {df_weekly['week'].max()-3}-{df_weekly['week'].max()} (Average: {late_avg:.1f} pts/game)
                    
                    We compare these two periods to identify how scoring patterns change throughout the season. 
                    Factors like weather, defensive improvements, and injuries often affect late-season scoring.
                    """)

    with tab2:
        # Home advantage over time
        fig_home = go.Figure()
        
        fig_home.add_trace(go.Scatter(
            x=df_weekly['week'],
            y=df_weekly['home_win_rate'] * 100,
            mode='lines+markers',
            name='Home Win %',
            line=dict(width=3, color='green'),
            marker=dict(size=10),
            fill='tonexty',
            fillcolor='rgba(0,255,0,0.1)'
        ))
        
        # Add 50% reference line
        fig_home.add_hline(
            y=50,
            line_dash="dash",
            line_color="gray",
            annotation_text="50% (No Advantage)",
            annotation_position="right"
        )
        
        fig_home.update_layout(
            title="Home Team Win Percentage by Week",
            xaxis_title="Week",
            yaxis_title="Home Win %",
            yaxis_range=[0, 100],
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_home, use_container_width=True)
        
        # Insights
        st.info(f"🏠 **Overall home advantage:** {overview_stats['home_win_pct']:.1%} (vs expected 50%)")
        
        # Find weeks with strongest/weakest home advantage
        strongest_home = df_weekly.loc[df_weekly['home_win_rate'].idxmax()]
        weakest_home = df_weekly.loc[df_weekly['home_win_rate'].idxmin()]
        
        col_a, col_b = st.columns(2)
        col_a.metric(
            "Strongest Home Advantage",
            f"Week {int(strongest_home['week'])}",
            f"{strongest_home['home_win_rate']:.1%}"
        )
        col_b.metric(
            "Weakest Home Advantage",
            f"Week {int(weakest_home['week'])}",
            f"{weakest_home['home_win_rate']:.1%}"
        )
    
    with tab3:
        # Competitiveness metrics
        fig_comp = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Avg margin (lower = more competitive)
        fig_comp.add_trace(
            go.Scatter(
                x=df_weekly['week'],
                y=df_weekly['avg_margin'],
                name="Avg Point Margin",
                mode='lines+markers',
                line=dict(color='orange', width=3),
                marker=dict(size=8)
            ),
            secondary_y=False
        )
        
        # Close game rate (higher = more competitive)
        fig_comp.add_trace(
            go.Scatter(
                x=df_weekly['week'],
                y=df_weekly['close_game_rate'] * 100,
                name="Close Game % (≤7 pts)",
                mode='lines+markers',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True
        )
        
        fig_comp.update_xaxes(title_text="Week")
        fig_comp.update_yaxes(title_text="Avg Point Margin", secondary_y=False)
        fig_comp.update_yaxes(title_text="Close Game %", secondary_y=True, range=[0, 100])
        fig_comp.update_layout(
            title="Game Competitiveness by Week",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Find most competitive week
        most_comp_week = df_weekly.loc[df_weekly['avg_margin'].idxmin()]
        most_close_week = df_weekly.loc[df_weekly['close_game_rate'].idxmax()]
        
        st.info(f"⚔️ **Most competitive week (smallest avg margin):** Week {int(most_comp_week['week'])} with {most_comp_week['avg_margin']:.1f} pt average margin")
        st.info(f"🎯 **Week with most close games:** Week {int(most_close_week['week'])} with {most_close_week['close_game_rate']:.1%} of games decided by ≤7 pts")
    
    st.markdown("---")
    
    # ========================================================================
    # 1-5) Game Distribution
    # ========================================================================
    st.markdown("### 📊 Game Statistics Distribution")
    st.caption("💡 View the distribution of key game-level metrics across all games this season")
    
    # Clean game distribution data
    for col in ['total_points', 'point_margin', 'home_score', 'away_score', 'home_total_yards', 'away_total_yards']:
        if col in df_game_dist.columns:
            df_game_dist[col] = pd.to_numeric(df_game_dist[col], errors='coerce')
    
    df_game_dist = df_game_dist.dropna(subset=['total_points', 'point_margin'])
    
    game_metrics = {
        'total_points': 'Total Points per Game',
        'point_margin': 'Point Differential',
        'home_score': 'Home Team Score',
        'away_score': 'Away Team Score',
        'home_total_yards': 'Home Total Yards',
        'away_total_yards': 'Away Total Yards',
    }
    
    metric_for_dist = st.selectbox(
        "Choose a game metric to visualize:",
        list(game_metrics.keys()),
        format_func=lambda x: game_metrics[x],
        key="game_dist_metric"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig_hist = px.histogram(
            df_game_dist,
            x=metric_for_dist,
            nbins=25,
            title=f"{game_metrics[metric_for_dist]} Distribution",
            labels={metric_for_dist: game_metrics[metric_for_dist]}
        )
        
        # Add median line
        median_val = df_game_dist[metric_for_dist].median()
        fig_hist.add_vline(
            x=median_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {median_val:.1f}",
            annotation_position="top right"
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot by week
        fig_box = px.box(
            df_game_dist,
            x='week',
            y=metric_for_dist,
            title=f"{game_metrics[metric_for_dist]} by Week",
            labels={
                'week': 'Week',
                metric_for_dist: game_metrics[metric_for_dist]
            }
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Summary statistics
    st.markdown("#### Summary Statistics")
    summary = df_game_dist[metric_for_dist].describe()
    cols = st.columns(6)
    cols[0].metric("Mean", f"{summary['mean']:.1f}")
    cols[1].metric("Median", f"{summary['50%']:.1f}")
    cols[2].metric("Std Dev", f"{summary['std']:.1f}")
    cols[3].metric("Min", f"{summary['min']:.1f}")
    cols[4].metric("Max", f"{summary['max']:.1f}")
    cols[5].metric("Range", f"{summary['max'] - summary['min']:.1f}")
    
    st.markdown("---")
    
    # ========================================================================
    # 1-6) Top Performers Table
    # ========================================================================
    st.markdown("### 🏅 Top & Bottom Teams by Metric")
    st.caption("💡 Team-level performance rankings based on season averages")
    
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
    
    col_top, col_bottom = st.columns(2)
    
    with col_top:
        st.markdown("#### 🏆 Top 10")
        df_top = df_pct[["team_name", m_col, m_col + "_pctile"]].dropna()
        df_top = df_top.sort_values(by=m_col, ascending=not higher_is_better).head(10)
        df_top.insert(0, 'Rank', range(1, len(df_top) + 1))
        
        st.dataframe(
            df_top.rename(
                columns={
                    "team_name": "Team",
                    m_col: metric_label(m_col),
                    m_col + "_pctile": "Percentile",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    
    with col_bottom:
        st.markdown("#### 📉 Bottom 10")
        df_bottom = df_pct[["team_name", m_col, m_col + "_pctile"]].dropna()
        df_bottom = df_bottom.sort_values(by=m_col, ascending=higher_is_better).head(10)
        df_bottom.insert(0, 'Rank', range(len(df_stats) - len(df_bottom) + 1, len(df_stats) + 1))
        
        st.dataframe(
            df_bottom.rename(
                columns={
                    "team_name": "Team",
                    m_col: metric_label(m_col),
                    m_col + "_pctile": "Percentile",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    
    st.markdown("---")
    
    # ========================================================================
    # 1-7) League Heatmap – team vs median
    # ========================================================================
    st.markdown("### 🔥 League Performance Heatmap (vs Median)")
    st.caption("💡 Compare each team's performance across all metrics relative to the league median. Green = above median performance, Red = below median. Each cell shows team performance relative to league median (1.0 = median)")
    
    # Add option to filter top N teams
    show_all = st.checkbox("Show all teams", value=False)
    
    if not show_all:
        top_n = st.slider(
            "Show top N teams (by win%)",
            min_value=20,
            max_value=len(df_stats),
            value=min(50, len(df_stats)),
            step=10
        )
        df_for_heat = df_stats.nlargest(top_n, 'win_pct')
    else:
        df_for_heat = df_stats
    
    heat_df = df_for_heat[["team_name"] + TEAM_METRICS].copy()
    for col in TEAM_METRICS:
        if HIGHER_IS_BETTER.get(col, True):
            heat_df[col] = heat_df[col] / benchmark[col]
        else:
            heat_df[col] = benchmark[col] / heat_df[col]
    
    heat_matrix = heat_df.set_index("team_name")[TEAM_METRICS]
    
    # Rename columns for better readability
    heat_matrix.columns = [metric_label(c) for c in TEAM_METRICS]
    
    heat_fig = px.imshow(
        heat_matrix,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        origin="upper",
        labels=dict(color="Relative to Median"),
        title=f"Team Performance vs League Median ({'Top ' + str(top_n) if not show_all else 'All'} Teams)",
        zmin=0.5,
        zmax=1.5
    )
    
    heat_fig.update_layout(height=max(400, len(heat_matrix) * 20))
    
    st.plotly_chart(heat_fig, use_container_width=True)
    
    with st.expander("📖 How to Read This Heatmap (Click to Expand)", expanded=False):
        st.markdown("""
        ### Understanding the Heatmap
        
        #### 🎨 **Color Coding**
        - 🟢 **Green cells**: Performance **above** league median (stronger)
        - 🟡 **Yellow cells**: Performance **near** league median (average)  
        - 🔴 **Red cells**: Performance **below** league median (weaker)
        
        #### 📊 **Team Ranking (Y-axis / Left Side)**
        - Teams are **sorted by win percentage**
        - Reading from **bottom to top**: worst win% → best win%
        - Teams at the **top** have the best records
        - Teams at the **bottom** have the worst records
        
        #### 📏 **Understanding the Values**
        Each cell contains a number representing **relative performance to median**:
        - **1.0** = Exactly at league median
        - **1.25** = 25% better than median
        - **0.75** = 25% worse than median
        - **Range**: 0.5 (very weak) to 1.5+ (very strong)
        
        #### 🔍 **How to Analyze**
        
        **Reading Horizontally (by Row):**
        - Shows one team's performance across all metrics
        - Example: If Team A has mostly green cells → well-rounded strong team
        - Example: If Team B has green offense but red defense → unbalanced team
        
        **Reading Vertically (by Column):**
        - Shows how all teams perform in one specific metric
        - Example: If a column is mostly red → this is a weak area league-wide
        - Example: If a column has mixed colors → high variance in this skill
        
        #### 💡 **Practical Examples**
        
        | Pattern | Meaning |
        |---------|---------|
        | 🟢🟢🟢🟢 (All green row) | Elite team, strong in all areas |
        | 🔴🔴🔴🔴 (All red row) | Struggling team, weak across the board |
        | 🟢🟢🔴🔴 (Mixed row) | Specialized team (e.g., good offense, poor defense) |
        | 🟢 vertical column | League-wide strength (e.g., everyone scores well) |
        | 🔴 vertical column | League-wide weakness (e.g., everyone struggles defensively) |
        
        #### 🎯 **Quick Tips**
        - **Hover** over any cell to see exact values
        - Look for **clusters of color** to identify team archetypes
        - Compare teams at **similar win%** to find hidden differences
        """)

    st.info("💡 **Quick Guide:** Values > 1.0 (green) indicate above-median performance. Values < 1.0 (red) indicate below-median performance.")

# ============================================================================
# 2. Team Performance
# ============================================================================
elif page == "2. Team Performance":
    st.subheader("🎯 Team Performance vs League Benchmark")
    
    st.markdown("""
    Analyze individual team performance across multiple dimensions, compare against league benchmarks,
    and track performance trends throughout the season.
    """)

    with st.spinner("Loading team stats & benchmark..."):
        df_stats, benchmark = load_team_stats_with_benchmark()
        df_pct = compute_percentiles(df_stats, TEAM_METRICS)

    # ========================================================================
    # Team Selection
    # ========================================================================
    st.markdown("### 🔍 Select a Team to Analyze")
    
    team_options = df_stats["team_name"].sort_values().unique().tolist()
    selected_team = st.selectbox("Choose a team", team_options, key="team_perf_select")

    team_row = df_pct[df_pct["team_name"] == selected_team].iloc[0]
    team_id = team_row['team_id']
    
    st.markdown("---")

     # ========================================================================
    # 2-1) Game Log
    # ========================================================================
    st.markdown("### 📋 Season Game Log")

    @st.cache_data(show_spinner="Loading game log...")
    def load_team_game_log(team_id, season):
        """Load all games for a specific team with opponent info and logos"""
       
        team_id = int(team_id)
        season = int(season)
        
        sql = """
            WITH team_games AS (
                SELECT 
                    pc.game_id,
                    g.week,
                    g.start_date,
                    CASE 
                        WHEN pc.home_team_id = ? THEN 'home'
                        ELSE 'away'
                    END AS home_away,
                    CASE 
                        WHEN pc.home_team_id = ? THEN pc.home_score
                        ELSE pc.away_score
                    END AS team_score,
                    CASE 
                        WHEN pc.home_team_id = ? THEN pc.away_score
                        ELSE pc.home_score
                    END AS opponent_score,
                    CASE 
                        WHEN pc.home_team_id = ? THEN pc.away_team_id
                        ELSE pc.home_team_id
                    END AS opponent_id,
                    CASE 
                        WHEN pc.home_team_id = ? THEN pc.home_total_yards
                        ELSE pc.away_total_yards
                    END AS team_yards,
                    CASE 
                        WHEN pc.home_team_id = ? THEN pc.home_turnovers
                        ELSE pc.away_turnovers
                    END AS team_turnovers
                FROM bt.pairwise_comparisons AS pc
                JOIN real_deal.dim_games AS g ON pc.game_id = g.id
                WHERE (pc.home_team_id = ? OR pc.away_team_id = ?)
                    AND g.season = ?
            )
            SELECT 
                tg.*,
                t.display_name AS opponent,
                t.logo AS opponent_logo
            FROM team_games tg
            JOIN real_deal.dim_teams AS t ON tg.opponent_id = t.id
            ORDER BY tg.start_date
        """
        df = run_query(sql, (team_id, team_id, team_id, team_id, team_id, team_id, team_id, team_id, season))
        df['result'] = df.apply(lambda row: 'W' if row['team_score'] > row['opponent_score'] else 'L', axis=1)
        df['location'] = df['home_away'].apply(lambda x: '🏠' if x == 'home' else '✈️')
        df['margin'] = df['team_score'] - df['opponent_score']
        return df

    df_games = load_team_game_log(team_id, SEASON)
    if len(df_games) > 0:
        # Get team logo
        team_logo = df_stats[df_stats['team_name'] == selected_team]['logo'].iloc[0] if 'logo' in df_stats.columns else None
        
        if team_logo is None or pd.isna(team_logo):
            logo_sql = """
                SELECT logo 
                FROM real_deal.dim_teams 
                WHERE id = ?
            """
            logo_result = run_query(logo_sql, (int(team_id),))
            if len(logo_result) > 0:
                team_logo = logo_result.iloc[0]['logo']
        
        # Game log summary with team logo
        
        col_logo, col1, col2, col3 = st.columns(4) 
        
        # Show team logo
        if team_logo and not pd.isna(team_logo):
            with col_logo:
                logo_html = f"""
                <div style="text-align: left;">
                <img src="{team_logo}" 
                    alt="{selected_team}" 
                    title="{selected_team}"
                    style="width: 80px; height: 80px; object-fit: contain; cursor: pointer;"
                    onmouseover="this.style.transform='scale(1.1)'; this.style.transition='transform 0.2s';"
                    onmouseout="this.style.transform='scale(1)';">
                </div>
                """
                st.markdown(logo_html, unsafe_allow_html=True)
        else:
            col_logo.markdown(f"### {selected_team[:3]}") 
        
        # Calculate Statistics
        wins = (df_games['result'] == 'W').sum()
        losses = (df_games['result'] == 'L').sum()
        home_games = df_games[df_games['home_away'] == 'home']
        away_games = df_games[df_games['home_away'] == 'away']
        home_wins = (home_games['result'] == 'W').sum() if len(home_games) > 0 else 0
        away_wins = (away_games['result'] == 'W').sum() if len(away_games) > 0 else 0
        
        col1.metric("Overall Record", f"{wins}-{losses}")
        col2.metric("Home Record", f"{home_wins}-{len(home_games) - home_wins}" if len(home_games) > 0 else "0-0")
        col3.metric("Away Record", f"{away_wins}-{len(away_games) - away_wins}" if len(away_games) > 0 else "0-0")
        
        # Display game log table with logos
        st.dataframe(
            df_games[[
                'week', 'start_date', 'result', 'location', 
                'opponent_logo', 'opponent', 'team_score', 'opponent_score', 'team_yards', 'team_turnovers'
            ]].rename(columns={             
                'week': 'Week',
                'start_date': 'Date',
                'result': 'W/L',
                'location': 'Loc',
                'opponent_logo': 'Opponent Logo',
                'opponent': 'Opponent',
                'team_score': 'Pts',
                'opponent_score': 'Opp Pts',
                'team_yards': 'Yards',
                'team_turnovers': 'Turnovers'
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Opponent Logo": st.column_config.ImageColumn(
                    width="small"  # small, medium, large
                ),
                "Date": st.column_config.DateColumn(format="MMM DD, YYYY"),
                "W/L": st.column_config.TextColumn(
                    help="W = Win, L = Loss"
                ),
                "Loc": st.column_config.TextColumn(
                    help="🏠 = Home, ✈️ = Away"
                ),
            }
        )
    else:
        st.info("No game data available for this team.")

    st.markdown("---")

    # ========================================================================
    # 2-2) Enhanced KPI Cards with Rankings + Bradley-Terry
    # ========================================================================
    st.markdown("### 📌 Key Performance Indicators")

    # Get Bradley-Terry Ranking
    @st.cache_data
    def get_bt_ranking(team_id):
        """Get Bradley-Terry ranking for a team"""
        sql = """
            SELECT 
                rank,
                strength,
                prob_vs_avg
            FROM bt.rankings
            WHERE team_id = ?
        """
        result = run_query(sql, (int(team_id),))
        if len(result) > 0:
            return result.iloc[0]
        return None

    bt_rank = get_bt_ranking(team_id)

    # 5 Columns KPI Cards
    kpi_cols = ["win_pct", "avg_points_scored", "avg_points_allowed", "point_differential"]
    kpi_display = [metric_label(c) for c in kpi_cols]

    cols = st.columns(5)

    # First Column: Bradley-Terry Ranking
    if bt_rank is not None:
        cols[0].metric(
            label="BT Ranking",
            value=f"#{int(bt_rank['rank'])}",
            delta=f"Strength: {bt_rank['strength']:.3f}",
            help=f"Bradley-Terry model ranking. Win probability vs average team: {bt_rank['prob_vs_avg']:.1%}"
        )
        cols[0].caption(f"💪 {bt_rank['prob_vs_avg']:.1%} vs avg team")
    else:
        cols[0].metric("BT Ranking", "N/A")

    # KPIs
    for i, (c, label) in enumerate(zip(kpi_cols, kpi_display), start=1):
        val = float(team_row[c])
        med = float(benchmark[c])
        delta_pct = (val - med) / med if med != 0 else 0.0
        
        # Calculate rank for this metric
        if HIGHER_IS_BETTER.get(c, True):
            rank = (df_stats[c] > val).sum() + 1
        else:
            rank = (df_stats[c] < val).sum() + 1
        
        cols[i].metric(
            label=f"{label}",
            value=f"{val:.3f}" if c == "win_pct" else f"{val:.1f}",
            delta=f"{delta_pct:+.1%} vs median",
            help=f"Rank: #{rank} out of {len(df_stats)} teams"
        )
        cols[i].caption(f"📊 Rank: #{rank}/{len(df_stats)}")

    st.markdown("---")

    # ========================================================================
    # 2-3) Strength of Schedule
    # ========================================================================
    st.markdown("### 💪 Strength of Schedule")
    
    @st.cache_data(show_spinner="Calculating strength of schedule...")
    def calculate_sos(team_id, season):
        """Calculate strength of schedule based on opponents' Bradley-Terry ratings"""
        team_id = int(team_id)
        season = int(season)

        sql = """
            WITH team_opponents AS (
                SELECT 
                    CASE 
                        WHEN pc.home_team_id = ? THEN pc.away_team_id
                        ELSE pc.home_team_id
                    END AS opponent_id,
                    CASE 
                        WHEN pc.home_team_id = ? THEN 'away'
                        ELSE 'home'
                    END AS opponent_location
                FROM bt.pairwise_comparisons AS pc
                JOIN real_deal.dim_games AS g ON pc.game_id = g.id
                WHERE (pc.home_team_id = ? OR pc.away_team_id = ?)
                    AND g.season = ?
            )
            SELECT 
                AVG(r.strength) AS avg_opponent_strength,
                AVG(CASE WHEN opp.opponent_location = 'home' THEN r.strength END) AS avg_home_opp_strength,
                AVG(CASE WHEN opp.opponent_location = 'away' THEN r.strength END) AS avg_away_opp_strength,
                COUNT(*) AS total_opponents
            FROM team_opponents opp
            JOIN bt.rankings AS r ON r.team_id = opp.opponent_id
        """
        result = run_query(sql, (team_id, team_id, team_id, team_id, season))
        if len(result) > 0:
            return result.iloc[0]
        return None
    
    sos = calculate_sos(team_id, SEASON)
    
    if sos is not None and sos['total_opponents'] > 0:
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "Overall SoS",
            f"{sos['avg_opponent_strength']:.3f}" if pd.notna(sos['avg_opponent_strength']) else "N/A",
            help="Average Bradley-Terry strength of all opponents"
        )
        col2.metric(
            "Home Opponents",
            f"{sos['avg_home_opp_strength']:.3f}" if pd.notna(sos['avg_home_opp_strength']) else "N/A",
            help="Average strength of opponents faced at home"
        )
        col3.metric(
            "Away Opponents",
            f"{sos['avg_away_opp_strength']:.3f}" if pd.notna(sos['avg_away_opp_strength']) else "N/A",
            help="Average strength of opponents faced on the road"
        )
        
        st.info("💡 **Interpretation:** Higher values indicate tougher opponents. League average strength is around 1.000. A SoS > 1.0 means above-average schedule difficulty.")
    else:
        st.info("Strength of schedule data not available (requires Bradley-Terry rankings).")

    st.markdown("---")

    # ========================================================================
    # 2-4) Performance Trends
    # ========================================================================
    st.markdown("### 📈 Season Performance Trends")
    
    if len(df_games) > 0:
        # Calculate cumulative and rolling statistics
        df_trends = df_games.sort_values('start_date').copy()
        df_trends['cumulative_wins'] = (df_trends['result'] == 'W').cumsum()
        df_trends['cumulative_games'] = range(1, len(df_trends) + 1)
        df_trends['win_pct_running'] = df_trends['cumulative_wins'] / df_trends['cumulative_games']
        
        # 5-game rolling averages
        df_trends['points_ma5'] = df_trends['team_score'].rolling(window=5, min_periods=1).mean()
        df_trends['points_allowed_ma5'] = df_trends['opponent_score'].rolling(window=5, min_periods=1).mean()
        df_trends['margin_ma5'] = df_trends['margin'].rolling(window=5, min_periods=1).mean()
        
        tab1, tab2, tab3 = st.tabs(["🏆 Win % Trend", "📊 Scoring Trends", "⚖️ Margin Trend"])
        
        with tab1:
            fig_winpct = px.line(
                df_trends,
                x='week',
                y='win_pct_running',
                title=f"{selected_team} - Cumulative Win % Over Season",
                labels={'week': 'Week', 'win_pct_running': 'Win %'},
                markers=True
            )
            fig_winpct.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="gray",
                annotation_text="50%"
            )
            fig_winpct.update_yaxes(range=[0, 1], tickformat='.0%')
            fig_winpct.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_winpct, use_container_width=True)
            
            # Insight
            if len(df_trends) >= 6:
                early_winpct = df_trends.iloc[:len(df_trends)//2]['win_pct_running'].iloc[-1]
                current_winpct = df_trends['win_pct_running'].iloc[-1]
                change = current_winpct - early_winpct

                if abs(change) > 0.1:
                    trend = "improved" if change > 0 else "declined"
                    icon = "📈" if change > 0 else "📉"
                    st.info(f"{icon} **Trend:** Win % has {trend} from {early_winpct:.1%} (mid-season) to {current_winpct:.1%} (current) - a change of {change:+.1%}")

            with st.expander("💡 How to Interpret Win % Trend"):
                    st.markdown("""
                    ### 📊 Understanding Win % Trend
                    
                    This chart shows the **cumulative win percentage** over the season. Each point represents the team's overall record up to that week.
                    
                    #### 🔍 What to Look For:
                    
                    **📈 Upward Trend (Improving)**
                    - Team is winning more games as season progresses
                    - Possible reasons: tactical adjustments, player development, injury recoveries, easier schedule
                    - **Positive sign** for playoffs
                    
                    **📉 Downward Trend (Declining)**
                    - Team is losing more games recently
                    - Possible reasons: key injuries, fatigue, opponents adapting to tactics, harder schedule
                    - **Warning sign** - investigate causes
                    
                    **➡️ Flat Trend (Stable)**
                    - Consistent performance throughout season
                    - If high (70%+): Strong, reliable team
                    - If low (30%-): Struggling team
                    
                    **🌊 High Volatility (Wavy line)**
                    - Inconsistent performance
                    - Team may be "match-up dependent" or unpredictable
                    
                    #### 💡 Pro Tip:
                    Compare with **Strength of Schedule** to understand if trend is due to team improvement or opponent difficulty changes.
                    
                    **Example:**
                    - Win % rising + SoS increasing = Team genuinely improving ✅
                    - Win % rising + SoS decreasing = Might be easier schedule ⚠️
                    """)

        with tab2:
            fig_scoring = go.Figure()
            fig_scoring.add_trace(go.Scatter(
                x=df_trends['week'],
                y=df_trends['points_ma5'],
                name='Points Scored (5-game avg)',
                mode='lines+markers',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ))
            fig_scoring.add_trace(go.Scatter(
                x=df_trends['week'],
                y=df_trends['points_allowed_ma5'],
                name='Points Allowed (5-game avg)',
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ))
            fig_scoring.update_layout(
                title=f"{selected_team} - Scoring Trends (5-game Moving Average)",
                xaxis_title="Week",
                yaxis_title="Points",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_scoring, use_container_width=True)
            
            # Recent form analysis
            if len(df_trends) >= 5:
                recent_5 = df_trends.tail(5)
                recent_ppg = recent_5['team_score'].mean()
                recent_papg = recent_5['opponent_score'].mean()
                st.info(f"📊 **Last 5 games:** Scoring {recent_ppg:.1f} PPG, Allowing {recent_papg:.1f} PPG")

            with st.expander("💡 How to Interpret Scoring Trends"):
                st.markdown("""
            ### 📊 Understanding Scoring Trends
            
            This chart shows **5-game moving averages** of points scored (green) and points allowed (red).
            
            #### Why 5-Game Moving Average?
            
            **Problem with raw game-by-game scores:**

            Week 1: 45 pts 
            Week 2: 28 pts  ← Big drop! Why? 
            Week 3: 52 pts  ← Big jump! Why?

            - Too much **noise** - single games affected by weather, opponent strength, randomness
            - Hard to see actual **trends**
            
            **Solution: 5-Game Moving Average**
            - Smooths out random fluctuations
            - Shows true performance trends
            - Represents "recent form" (~1 month of play)
            - **Balance:** Responsive enough to catch changes, stable enough to filter noise
            
            #### 🔍 What to Look For:
            
            **📈 Green Line (Points Scored) Rising**
            - Offense improving!
            - Possible causes: QB finding rhythm, O-line gelling, play-calling improvements
            
            **📉 Green Line Falling**
            - Offense struggling
            - Check: Key injuries? Opponents' defenses getting stronger? (see SoS)
            
            **📉 Red Line (Points Allowed) Falling** ✅
            - Defense improving! (Lower is better)
            - Possible causes: Defensive adjustments working, players adapting to scheme
            
            **📈 Red Line Rising** ❌
            - Defense struggling
            - Check: Defensive injuries? Fatigue? Stronger opponents?
            
            #### 🎯 Line Relationship Patterns:
            
            **1. Scissors Pattern (Ideal)** 🔥

            Green ↗ (Offense improving) 
            Red ↘ (Defense improving)

            → Both sides of ball getting better!
            
            **2. Reverse Scissors (Danger)** ⚠️

            Green ↘ (Offense declining) 
            Red ↗ (Defense declining)

            → Team falling apart
            
            **3. Both Lines High**

            Green: 35+ pts 
            Red: 28+ pts

            → High-tempo, shootout style (exciting but risky)
            
            **4. Both Lines Low**

            Green: 20- pts 
            Red: 15- pts

            → Defensive, low-scoring grind-it-out style
            
            **5. Crossover Point** ✂️

            Before crossover: Offense > Defense 
            After crossover: Defense > Offense

            → Team identity shift!
            
            #### 💡 Pro Tips:
            
            - **Gap between lines = Point margin**
              - Wider gap = Dominant team
              - Narrow gap = Close games
            
            - **Compare to KPIs:**
              - Does trend match avg points scored/allowed?
              - Recent trend might differ from season average
            
            - **Context matters:**
              - Check SoS: Are opponents getting harder/easier?
              - Check Game Log: Any pattern in wins/losses?
            """)

        with tab3:
            fig_margin = px.line(
                df_trends,
                x='week',
                y='margin_ma5',
                title=f"{selected_team} - Point Margin Trend (5-game Moving Average)",
                labels={'week': 'Week', 'margin_ma5': 'Point Margin'},
                markers=True
            )
            fig_margin.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                annotation_text="Even"
            )
            fig_margin.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_margin, use_container_width=True)

            with st.expander("💡 How to Interpret Margin Trend"):
                st.markdown("""
            ### 📊 Understanding Point Margin Trend
            
            This chart shows the **5-game moving average** of point differential (team score - opponent score).
            
            #### 📏 Margin Scale:
            
            | Avg Margin | Rating | Meaning |
            |------------|--------|---------|
            | **+15 or more** | 🔥 Dominant | Crushing opponents |
            | **+10 to +15** | 💪 Strong | Winning decisively |
            | **+5 to +10** | ✅ Solid | Winning comfortably |
            | **+0 to +5** | 😐 Close wins | Winning by small margins (risky) |
            | **-0 to -5** | ⚠️ Close losses | Losing by small margins (close to winning) |
            | **-5 to -10** | 📉 Weak | Losing regularly |
            | **-10 or worse** | 💔 Dominated | Getting blown out |
            
            #### 🔍 What to Look For:
            
            **📈 Margin Widening (ex: +5 → +12)**
            - Team is winning by larger margins
            - Possible reasons:
              - Team getting stronger
              - Better execution
              - Improved chemistry
              - *Check SoS:* Or are opponents getting weaker?
            
            **📉 Margin Narrowing (ex: +12 → +5)**
            - Games becoming closer
            - Possible reasons:
              - Team strength declining
              - Opponents getting stronger
              - Key injuries
              - Fatigue
            
            **🔄 Margin Flips Positive (ex: -3 → +6)**
            - Team "turned the corner"!
            - Major improvement or adjustment
            - **Good sign** for playoffs
            
            **🔄 Margin Flips Negative (ex: +8 → -2)**
            - Team collapse
            - Something went seriously wrong
            - **Major red flag**
            
            #### 📊 Relationship with Zero Line:
            
            **Always Above 0** ✅

            → Winning consistently all season
            
            **Hovering Around 0** ⚖️

            → Close games, 50-50 team, unpredictable
            
            **Always Below 0** ❌

            → Getting beaten all season
            
            #### 🎯 Cross-Validation with Other Tabs:
            
            **Scenario 1: Margin widening, but scoring down**

            Points Scored:  35 → 30 (↓)
            Points Allowed: 28 → 18 (↓)
            Margin:         +7 → +12 (↑)

            **Interpretation:** Not offense getting better - it's **defense improving**!
            Team transformed into defensive squad.
            
            **Scenario 2: Margin shrinking, but scoring up**

            Points Scored:  28 → 35 (↑)
            Points Allowed: 21 → 32 (↑)
            Margin:         +7 → +3 (↓)

            **Interpretation:** Offense better but **defense collapsed**! Danger sign!
            
            **Scenario 3: Margin stable, both scores up**

            Points Scored:  28 → 38 (↑)
            Points Allowed: 21 → 31 (↑)
            Margin:         +7 → +7 (=)

            **Interpretation:** Pace of play increased. Now playing **shootout style**.
            
            #### 💡 Pro Tips:
            
            - **Margin = Quality of wins/losses**
              - +15 margin = Dominant wins (crushing opponents)
              - +3 margin = Lucky wins (could easily lose)
            
            - **Trend = Trajectory**
              - Rising margin = Team improving, peaking at right time
              - Falling margin = Team declining, might miss playoffs
            
            - **Volatility = Consistency**
              - Smooth line = Predictable team
              - Jagged line = Unpredictable, match-up dependent
            
            - **Always check context:**
              - SoS: Are opponents getting harder/easier?
              - Injuries: Did key players get hurt/return?
              - Game Log: Win streak? Losing streak?
            """)
    else:
        st.info("Not enough game data to show trends.")

    st.markdown("---")

    # ========================================================================
    # 2-5) Radar Chart with Standardized Metrics (0-1 scale)
    # ========================================================================
    st.markdown("### 🕸 Multi-dimensional Performance Radar")

    radar_metrics = [
        "win_pct",
        "avg_points_scored",
        "avg_points_allowed",
        "avg_total_yards",
        "avg_yards_allowed",
        "turnover_margin",
    ]

    @st.cache_data
    def calculate_standardized_radar_data(df_stats, selected_team, metrics):
        """
        Standardize all metrics to 0-1 scale, then calculate median and team values.
        For metrics where lower is better, invert the scale.
        """
        standardized_data = {}
        
        for metric in metrics:
            values = df_stats[metric].copy()
            
            # Get min and max
            min_val = values.min()
            max_val = values.max()
            
            # Standardize to 0-1
            if max_val != min_val:
                standardized = (values - min_val) / (max_val - min_val)
            else:
                standardized = pd.Series([0.5] * len(values), index=values.index)
            
            # If lower is better, invert the scale
            if not HIGHER_IS_BETTER.get(metric, True):
                standardized = 1 - standardized
            
            standardized_data[metric] = standardized
        
        # Create DataFrame with standardized values
        std_df = pd.DataFrame(standardized_data)
        std_df['team_name'] = df_stats['team_name'].values
        
        # Calculate median for each metric
        median_values = std_df[metrics].median()
        
        # Get team values
        team_values = std_df[std_df['team_name'] == selected_team][metrics].iloc[0]
        
        return team_values.tolist(), median_values.tolist()

    team_radar_values, median_radar_values = calculate_standardized_radar_data(
        df_stats, selected_team, radar_metrics
    )

    labels = [metric_label(c) for c in radar_metrics]

    radar_fig = go.Figure()

    # Add team trace
    radar_fig.add_trace(
        go.Scatterpolar(
            r=team_radar_values,
            theta=labels,
            fill="toself",
            name=selected_team,
            line=dict(color='blue', width=2),
            fillcolor='rgba(0, 0, 255, 0.2)'
        )
    )

    # Add median trace
    radar_fig.add_trace(
        go.Scatterpolar(
            r=median_radar_values,
            theta=labels,
            fill="toself",
            name="League Median",
            line=dict(color='gray', width=2, dash='dash'),
            fillcolor='rgba(128, 128, 128, 0.1)'
        )
    )

    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1], 
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=['0', '0.25', '0.5', '0.75', '1.0']
            )
        ),
        showlegend=True,
        height=500
    )

    st.plotly_chart(radar_fig, use_container_width=True)

    st.caption("💡 **How to read:** All metrics are standardized to 0-1 scale (0 = worst in league, 1 = best in league). For defensive metrics (points/yards allowed), the scale is inverted so that higher values are better. The dashed gray line shows the league median (typically around 0.5).")

    st.markdown("---")

    # ========================================================================
    # 2-6) Bar chart – detailed comparison
    # ========================================================================
    st.markdown("### 📊 Detailed Metric Comparison")

    metric_multi = st.multiselect(
        "Select metrics to compare (max 6 recommended):",
        TEAM_METRICS,
        default=["avg_points_scored", "avg_points_allowed", "win_pct", "point_differential"],
        format_func=metric_label,
    )
    
    if len(metric_multi) > 8:
        st.warning("⚠️ Too many metrics selected. Chart may be crowded. Consider selecting fewer metrics.")

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
        text="value"
    )
    bar_fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    bar_fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("---")

    # ========================================================================
    # 2-7) Enhanced Percentile Table with Rankings
    # ========================================================================
    st.markdown("### 🪜 Complete Metric Rankings")

    # Calculate rankings for all metrics
    table_rows = []
    for m in TEAM_METRICS:
        val = float(team_row[m])
        pct = float(team_row[m + "_pctile"])
        
        # Calculate rank
        if HIGHER_IS_BETTER.get(m, True):
            rank = (df_stats[m] > val).sum() + 1
        else:
            rank = (df_stats[m] < val).sum() + 1
        
        # Determine grade
        if pct >= 75:
            grade = "🟢 Excellent"
        elif pct >= 50:
            grade = "🟡 Above Avg"
        elif pct >= 25:
            grade = "🟠 Below Avg"
        else:
            grade = "🔴 Poor"
        
        table_rows.append({
            "Metric": metric_label(m),
            "Value": val,
            "Rank": f"#{rank}",
            "Percentile": f"{pct:.1f}",
            "Grade": grade,
        })
    
    pct_df = pd.DataFrame(table_rows)
    
    # Display by category
    st.markdown("#### ⚔️ Offensive Metrics")
    offensive = ["Avg Points Scored", "Avg Total Yards", "Yards per Pass", "Yards per Rush", "Points per Yard (Offense)"]
    st.dataframe(
        pct_df[pct_df['Metric'].isin(offensive)], 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Value": st.column_config.NumberColumn(format="%.2f"),
            "Percentile": st.column_config.NumberColumn(format="%.1f"),
        }
    )
    
    st.markdown("#### 🛡️ Defensive Metrics")
    defensive = ["Avg Points Allowed", "Avg Yards Allowed", "Opp 3rd Down Eff", "Opp 4th Down Eff", "Points per Yard (Defense)"]
    st.dataframe(
        pct_df[pct_df['Metric'].isin(defensive)], 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Value": st.column_config.NumberColumn(format="%.2f"),
            "Percentile": st.column_config.NumberColumn(format="%.1f"),
        }
    )
    
    st.markdown("#### 🏆 Overall Performance")
    overall = ["Win %", "Point Differential", "Turnover Margin"]
    st.dataframe(
        pct_df[pct_df['Metric'].isin(overall)], 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Value": st.column_config.NumberColumn(format="%.2f"),
            "Percentile": st.column_config.NumberColumn(format="%.1f"),
        }
    )

    st.markdown("---")

    # ========================================================================
    # 2-8) Performance Gauges
    # ========================================================================
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

    st.markdown("""
    Explore team rankings from our Bradley-Terry model, compare with traditional polls (AP & Coaches),
    and read AI-generated team analyses.
    """)

    @st.cache_data(show_spinner="Loading all rankings...")
    def load_all_rankings_data(season):
        """
        Load Bradley-Terry, AP Poll, and Coaches Poll rankings
        """
        # 1. Bradley-Terry (current)
        bt_sql = """
            SELECT
                r.team_id,
                COALESCE(t.display_name, t.name) AS team_name,
                t.logo,
                r.rank AS bt_rank,
                r.strength AS bt_strength,
                r.prob_vs_avg
            FROM bt.rankings AS r
            LEFT JOIN real_deal.dim_teams AS t ON r.team_id = t.id
            ORDER BY r.rank
        """
        df_bt = run_query(bt_sql)
        
        # 2. Get latest week for polls
        week_sql = """
            SELECT MAX(week_number) as latest_week
            FROM real_deal.fact_rankings
            WHERE season_year = ?
        """
        week_result = run_query(week_sql, (season,))
        latest_week = int(week_result.iloc[0]['latest_week']) if len(week_result) > 0 and pd.notna(week_result.iloc[0]['latest_week']) else None
        
        df_ap = pd.DataFrame()
        df_coaches = pd.DataFrame()
        
        if latest_week is not None:
            # 3. AP Poll
            ap_sql = """
                SELECT
                    team_id,
                    team AS team_name,
                    current_rank AS ap_rank,
                    previous_rank AS ap_prev_rank,
                    points AS ap_points,
                    firstPlaceVotes AS ap_first_place_votes
                FROM real_deal.fact_rankings
                WHERE season_year = ?
                    AND week_number = ?
                    AND poll_name = 'AP Poll'
                ORDER BY current_rank
            """
            df_ap = run_query(ap_sql, (season, latest_week))
            
            # 4. Coaches Poll
            coaches_sql = """
                SELECT
                    team_id,
                    team AS team_name,
                    current_rank AS coaches_rank,
                    previous_rank AS coaches_prev_rank,
                    points AS coaches_points,
                    firstPlaceVotes AS coaches_first_place_votes
                FROM real_deal.fact_rankings
                WHERE season_year = ?
                    AND week_number = ?
                    AND poll_name = 'AFCA Coaches Poll'
                ORDER BY current_rank
            """
            df_coaches = run_query(coaches_sql, (season, latest_week))
        
        # 5. Merge all rankings
        if not df_ap.empty:
            df_merged = df_bt.merge(df_ap[['team_id', 'ap_rank', 'ap_prev_rank', 'ap_points', 'ap_first_place_votes']], 
                                    on='team_id', how='outer')
        else:
            df_merged = df_bt.copy()
            df_merged['ap_rank'] = None
            df_merged['ap_prev_rank'] = None
            df_merged['ap_points'] = None
            df_merged['ap_first_place_votes'] = None
        
        if not df_coaches.empty:
            df_merged = df_merged.merge(df_coaches[['team_id', 'coaches_rank', 'coaches_prev_rank', 'coaches_points', 'coaches_first_place_votes']], 
                                       on='team_id', how='outer')
        else:
            df_merged['coaches_rank'] = None
            df_merged['coaches_prev_rank'] = None
            df_merged['coaches_points'] = None
            df_merged['coaches_first_place_votes'] = None
        
        # Calculate consensus rank (average of available rankings)
        rank_cols = ['bt_rank', 'ap_rank', 'coaches_rank']
        df_merged['avg_rank'] = df_merged[rank_cols].mean(axis=1, skipna=True)
        df_merged['num_polls_ranked'] = df_merged[rank_cols].notna().sum(axis=1)
        
        return df_merged, latest_week
    
    @st.cache_data(show_spinner="Loading team summary...")
    def load_team_summary(team_id):
        """Load LLM-generated team summary from bt.team_summaries"""
        sql = """
            SELECT
                team_id,
                summary,
                updated_at
            FROM bt.team_summaries
            WHERE team_id = ?
        """
        result = run_query(sql, (int(team_id),))
        if len(result) > 0:
            return result.iloc[0]
        return None
    
    with st.spinner("Loading rankings & team stats..."):
        df_all_rankings, latest_week = load_all_rankings_data(SEASON)
        df_stats, benchmark = load_team_stats_with_benchmark()

    # Check if polls data is available
    has_ap = df_all_rankings['ap_rank'].notna().any()
    has_coaches = df_all_rankings['coaches_rank'].notna().any()

    # ========================================================================
    # 3-1) Current Rankings Dashboard with AI Summaries
    # ========================================================================
    st.markdown("### 🏆 Current Bradley-Terry Rankings")
    st.caption("💡 Our statistical model ranks teams based on game results and strength of schedule")
    
    # Top section: Rank selector + Team summary
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### 🔍 Select Rank")
        
        # Rank selector (1-25)
        selected_rank = st.selectbox(
            "Choose a rank to view team details:",
            range(1, 26),
            format_func=lambda x: f"#{x}",
            key="rank_selector"
        )
        
        # Get team at selected rank
        if len(df_all_rankings) >= selected_rank:
            selected_team_data = df_all_rankings[df_all_rankings['bt_rank'] == selected_rank].iloc[0]
            selected_team_name = selected_team_data['team_name']
            selected_team_id = selected_team_data['team_id']
            selected_strength = selected_team_data['bt_strength']
            selected_prob = selected_team_data['prob_vs_avg']
            selected_logo = selected_team_data.get('logo', None)
            
            # Display team info card
            st.markdown(f"### #{selected_rank}")
            # Team name with logo
            if selected_logo and pd.notna(selected_logo):
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px;">
                    <img src="{selected_logo}" width="30" height="30" style="border-radius: 5px;">
                    <span style="font-size: 20px; font-weight: bold;">{selected_team_name}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"**{selected_team_name}**")
            st.metric("Strength", f"{selected_strength:.3f}")
            st.metric("Win Prob vs Avg", f"{selected_prob:.1%}")
            
            # Get team stats
            team_stats = df_stats[df_stats['team_id'] == selected_team_id]
            if len(team_stats) > 0:
                team_record = team_stats.iloc[0]
                wins = int(team_record['wins'])
                losses = int(team_record['losses'])
                st.caption(f"**Record:** {wins}-{losses}")
        else:
            st.warning("Rank not available")
            selected_team_id = None
    
    with col2:
        st.markdown("#### 🤖 AI Team Analysis")
        
        if selected_team_id is not None:
            # Load AI summary
            @st.cache_data(show_spinner="Loading AI analysis...")
            def load_team_summary(team_id):
                """Load LLM-generated team summary from bt.team_summaries"""
                sql = """
                    SELECT
                        ts.team_id,
                        ts.summary,
                        r.updated_at
                    FROM bt.team_summaries AS ts
                    LEFT JOIN bt.rankings AS r ON ts.team_id = r.team_id
                    WHERE ts.team_id = ?
                """
                result = run_query(sql, (int(team_id),))
                if len(result) > 0:
                    return result.iloc[0]
                return None
            
            team_summary = load_team_summary(selected_team_id)
            
            if team_summary is not None:
                # Display summary in an attractive format
                st.markdown(f"""
                <div style="
                    background-color: var(--background-color, #f0f2f6); 
                    color: var(--text-color, #262730);
                    padding: 20px; 
                    border-radius: 10px; 
                    border-left: 5px solid var(--primary-color, #1f77b4);
                    min-height: 200px;
                ">
                    <h4 style="margin-top: 0; color: inherit;">📊 {selected_team_name}</h4>
                    <p style="font-size: 16px; line-height: 1.6; color: inherit;">
                        {team_summary['summary']}
                    </p>
                    <p style="font-size: 12px; color: var(--secondary-text-color, #666); margin-bottom: 0;">
                        <em>Analysis generated: {team_summary['updated_at'].strftime('%B %d, %Y')}</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick navigation buttons
                st.markdown("##### 🔗 Quick Actions")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("📋 View Full Stats", key="view_stats"):
                        st.info(f"💡 Go to **Team Performance** section and select '{selected_team_name}' to see detailed analytics")
                with col_b:
                    if st.button("📈 View Trends", key="view_trends"):
                        st.info(f"💡 Scroll down to see ranking history for '{selected_team_name}'")
                with col_c:
                    if st.button("⚔️ Compare", key="compare_team"):
                        st.info(f"💡 Go to **Head-to-Head** section to compare '{selected_team_name}' with other teams")
                
            else:
                st.info(f"""
                ### No AI Analysis Available
                
                AI-generated analysis is not yet available for **{selected_team_name}** (Rank #{selected_rank}).
                
                **Why?**
                - Analysis is only generated for Top 25 teams
                - Data may still be processing
                
                **What you can do:**
                - View detailed statistics in the **Team Performance** section
                - Check ranking history below
                - Compare with other teams in **Head-to-Head**
                """)
        else:
            st.info("Select a rank from 1-25 to view AI-generated team analysis")
    
    st.markdown("---")
    
    # ========================================================================
    # 3-2) Top 25 Rankings Tables (4 Tabs)
    # ========================================================================
    st.markdown("### 📊 Top 25 Rankings")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔵 Bradley-Terry", 
        "🔴 AP Poll", 
        "🟢 Coaches Poll", 
        "🏆 Combined Rankings"
    ])
    
    # Helper function to get teams with AI summaries
    @st.cache_data
    def get_teams_with_summaries():
        sql = "SELECT DISTINCT team_id FROM bt.team_summaries"
        result = run_query(sql)
        return set(result['team_id'].tolist())
    
    teams_with_ai = get_teams_with_summaries()
    
    # ====================================================================
    # TAB 1: Bradley-Terry Rankings (Original)
    # ====================================================================
    with tab1:
        st.markdown("#### Top 25 Teams - Bradley-Terry Rankings")
        st.caption("Statistical model based on game results and strength of schedule")
        
        # Get top 25 by Bradley-Terry
        df_bt_top25 = df_all_rankings[df_all_rankings['bt_rank'].notna()].sort_values('bt_rank').head(25).copy()
        
        # Merge with team stats
        df_bt_top25 = df_bt_top25.merge(
            df_stats[['team_id', 'wins', 'losses', 'win_pct', 'point_differential', 'avg_points_scored', 'avg_points_allowed']], 
            on='team_id', 
            how='left'
        )
        
        # Add AI summary indicator
        df_bt_top25['has_ai'] = df_bt_top25['team_id'].isin(teams_with_ai)
        
        # Create display dataframe
        display_df = df_bt_top25[[
            'bt_rank', 'team_name', 'bt_strength', 'wins', 'losses', 'win_pct', 
            'point_differential', 'avg_points_scored', 'avg_points_allowed', 'has_ai'
        ]].copy()
        
        st.dataframe(
            display_df.rename(columns={
                'bt_rank': 'Rank',
                'team_name': 'Team',
                'bt_strength': 'B-T Strength',
                'wins': 'W',
                'losses': 'L',
                'win_pct': 'Win %',
                'point_differential': 'Pt Diff',
                'avg_points_scored': 'Avg Pts',
                'avg_points_allowed': 'Avg PA',
                'has_ai': 'AI'
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(format="%d"),
                "B-T Strength": st.column_config.NumberColumn(format="%.3f"),
                "W": st.column_config.NumberColumn(format="%d"),
                "L": st.column_config.NumberColumn(format="%d"),
                "Win %": st.column_config.NumberColumn(format="%.3f"),
                "Pt Diff": st.column_config.NumberColumn(format="%+.1f"),
                "Avg Pts": st.column_config.NumberColumn(format="%.1f"),
                "Avg PA": st.column_config.NumberColumn(format="%.1f"),
                "AI": st.column_config.CheckboxColumn("AI Analysis"),
            },
            height=600
        )
        
        st.caption("✅ = AI analysis available | Select rank above to read analysis")
        
        # Show expandable full rankings
        with st.expander("📋 View Full Rankings (All Teams)"):
            df_all_bt = df_all_rankings[df_all_rankings['bt_rank'].notna()].sort_values('bt_rank')
            df_all_bt = df_all_bt.merge(
                df_stats[['team_id', 'wins', 'losses', 'win_pct']], 
                on='team_id', 
                how='left'
            )
            
            st.dataframe(
                df_all_bt[['bt_rank', 'team_name', 'bt_strength', 'prob_vs_avg', 'wins', 'losses', 'win_pct']].rename(columns={
                    'bt_rank': 'Rank',
                    'team_name': 'Team',
                    'bt_strength': 'Strength',
                    'prob_vs_avg': 'Win % vs Avg',
                    'wins': 'W',
                    'losses': 'L',
                    'win_pct': 'Win %'
                }),
                use_container_width=True,
                hide_index=True,
                height=400
            )
    
    # ====================================================================
    # TAB 2: AP Poll
    # ====================================================================
    with tab2:
        if has_ap:
            st.markdown("#### AP Top 25 (Associated Press Poll)")
            st.caption(f"Week {latest_week} rankings voted by sports writers and broadcasters")
            
            df_ap_display = df_all_rankings[df_all_rankings['ap_rank'].notna()].sort_values('ap_rank').head(25).copy()
            df_ap_display = df_ap_display.merge(
                df_stats[['team_id', 'wins', 'losses', 'win_pct']], 
                on='team_id', 
                how='left'
            )
            
            # Calculate rank change
            df_ap_display['ap_change'] = df_ap_display.apply(
                lambda row: int(row['ap_prev_rank'] - row['ap_rank']) if pd.notna(row['ap_prev_rank']) else 0,
                axis=1
            )
            
            # Add comparison with B-T
            df_ap_display['bt_diff'] = df_ap_display.apply(
                lambda row: int(row['bt_rank'] - row['ap_rank']) if pd.notna(row['bt_rank']) else None,
                axis=1
            )
            
            st.dataframe(
                df_ap_display[['ap_rank', 'team_name', 'ap_points', 'ap_first_place_votes', 
                              'ap_change', 'bt_rank', 'bt_diff', 'wins', 'losses', 'win_pct']].rename(columns={
                    'ap_rank': 'AP Rank',
                    'team_name': 'Team',
                    'ap_points': 'Points',
                    'ap_first_place_votes': '#1 Votes',
                    'ap_change': 'Δ',
                    'bt_rank': 'B-T Rank',
                    'bt_diff': 'vs B-T',
                    'wins': 'W',
                    'losses': 'L',
                    'win_pct': 'Win %'
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "AP Rank": st.column_config.NumberColumn(format="%d"),
                    "Points": st.column_config.NumberColumn(format="%d"),
                    "#1 Votes": st.column_config.NumberColumn(format="%d"),
                    "Δ": st.column_config.NumberColumn(format="%+d", help="Change from last week"),
                    "B-T Rank": st.column_config.NumberColumn(format="%d"),
                    "vs B-T": st.column_config.NumberColumn(format="%+d", help="Difference: B-T rank - AP rank (negative = ranked higher in B-T)"),
                    "W": st.column_config.NumberColumn(format="%d"),
                    "L": st.column_config.NumberColumn(format="%d"),
                    "Win %": st.column_config.NumberColumn(format="%.3f"),
                },
                height=600
            )
            
            st.caption("Δ = Change from last week | vs B-T = Difference from Bradley-Terry ranking")
            
            with st.expander("📚 About AP Poll"):
                st.markdown("""
                ### 📰 Associated Press Top 25 Poll
                
                **What it is:**
                - Voted by panel of 62 sports writers and broadcasters
                - Published weekly during the season
                - One of the most prestigious college football rankings
                
                **How voting works:**
                - Each voter ranks their top 25 teams
                - 1st place = 25 points, 2nd = 24 points, ..., 25th = 1 point
                - Total points determine final rankings
                - First-place votes indicate how many voters ranked a team #1
                
                **Strengths:**
                - Diverse perspectives from national media
                - Considers "eye test" and quality of play
                - Quick to react to big wins/losses
                
                **Limitations:**
                - Subject to voter bias and regional preferences
                - Voters may not watch every game
                - Hype and brand names can influence votes
                - Historical success can create inertia in rankings
                """)
        else:
            st.info("AP Poll data not available for current season")
    
    # ====================================================================
    # TAB 3: Coaches Poll
    # ====================================================================
    with tab3:
        if has_coaches:
            st.markdown("#### Coaches Top 25 (AFCA Coaches Poll)")
            st.caption(f"Week {latest_week} rankings voted by FBS head coaches")
            
            df_coaches_display = df_all_rankings[df_all_rankings['coaches_rank'].notna()].sort_values('coaches_rank').head(25).copy()
            df_coaches_display = df_coaches_display.merge(
                df_stats[['team_id', 'wins', 'losses', 'win_pct']], 
                on='team_id', 
                how='left'
            )
            
            # Calculate rank change
            df_coaches_display['coaches_change'] = df_coaches_display.apply(
                lambda row: int(row['coaches_prev_rank'] - row['coaches_rank']) if pd.notna(row['coaches_prev_rank']) else 0,
                axis=1
            )
            
            # Add comparison with B-T
            df_coaches_display['bt_diff'] = df_coaches_display.apply(
                lambda row: int(row['bt_rank'] - row['coaches_rank']) if pd.notna(row['bt_rank']) else None,
                axis=1
            )
            
            st.dataframe(
                df_coaches_display[['coaches_rank', 'team_name', 'coaches_points', 'coaches_first_place_votes', 
                                   'coaches_change', 'bt_rank', 'bt_diff', 'wins', 'losses', 'win_pct']].rename(columns={
                    'coaches_rank': 'Coaches Rank',
                    'team_name': 'Team',
                    'coaches_points': 'Points',
                    'coaches_first_place_votes': '#1 Votes',
                    'coaches_change': 'Δ',
                    'bt_rank': 'B-T Rank',
                    'bt_diff': 'vs B-T',
                    'wins': 'W',
                    'losses': 'L',
                    'win_pct': 'Win %'
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Coaches Rank": st.column_config.NumberColumn(format="%d"),
                    "Points": st.column_config.NumberColumn(format="%d"),
                    "#1 Votes": st.column_config.NumberColumn(format="%d"),
                    "Δ": st.column_config.NumberColumn(format="%+d", help="Change from last week"),
                    "B-T Rank": st.column_config.NumberColumn(format="%d"),
                    "vs B-T": st.column_config.NumberColumn(format="%+d", help="Difference: B-T rank - Coaches rank (negative = ranked higher in B-T)"),
                    "W": st.column_config.NumberColumn(format="%d"),
                    "L": st.column_config.NumberColumn(format="%d"),
                    "Win %": st.column_config.NumberColumn(format="%.3f"),
                },
                height=600
            )
            
            st.caption("Δ = Change from last week | vs B-T = Difference from Bradley-Terry ranking")
            
            with st.expander("📚 About Coaches Poll"):
                st.markdown("""
                ### 🎓 AFCA Coaches Poll
                
                **What it is:**
                - Voted by panel of FBS head coaches
                - Administered by American Football Coaches Association (AFCA)
                - One of the "major" polls alongside AP Poll
                
                **How voting works:**
                - Similar point system to AP Poll
                - 1st place = 25 points, descending to 1 point for 25th
                - Coaches cannot vote for their own team
                
                **Strengths:**
                - Expert opinion from people who study film and game-plan
                - Deep understanding of X's and O's
                - Insider knowledge of team strengths/weaknesses
                
                **Limitations:**
                - Coaches have limited time to watch other games
                - May vote strategically (helping/hurting rivals, conference)
                - Potential conflicts of interest
                - Often closely mirrors AP Poll
                """)
        else:
            st.info("Coaches Poll data not available for current season")
    
    # ====================================================================
    # TAB 4: Combined Rankings
    # ====================================================================
    with tab4:
        st.markdown("#### Combined Rankings (All Three Systems)")
        st.caption("Teams ranked in top 25 by ANY system")
        
        # Get teams ranked in top 25 by ANY system
        df_combined = df_all_rankings[
            (df_all_rankings['bt_rank'] <= 25) | 
            (df_all_rankings['ap_rank'] <= 25) | 
            (df_all_rankings['coaches_rank'] <= 25)
        ].copy()
        
        # Sort by average rank
        df_combined = df_combined.sort_values('avg_rank')
        
        # Merge with team stats
        df_combined = df_combined.merge(
            df_stats[['team_id', 'wins', 'losses', 'win_pct']], 
            on='team_id', 
            how='left'
        )
        
        # Add AI indicator
        df_combined['has_ai'] = df_combined['team_id'].isin(teams_with_ai)
        
        # Calculate rank spread
        df_combined['rank_spread'] = df_combined.apply(
            lambda row: (
                max([x for x in [row['bt_rank'], row['ap_rank'], row['coaches_rank']] if pd.notna(x)]) -
                min([x for x in [row['bt_rank'], row['ap_rank'], row['coaches_rank']] if pd.notna(x)])
            ) if row['num_polls_ranked'] >= 2 else 0,
            axis=1
        )
        
        st.dataframe(
            df_combined[['avg_rank', 'team_name', 'bt_rank', 'ap_rank', 'coaches_rank', 'rank_spread',
                        'wins', 'losses', 'win_pct', 'has_ai']].rename(columns={
                'avg_rank': 'Avg Rank',
                'team_name': 'Team',
                'bt_rank': 'B-T',
                'ap_rank': 'AP',
                'coaches_rank': 'Coaches',
                'rank_spread': 'Spread',
                'wins': 'W',
                'losses': 'L',
                'win_pct': 'Win %',
                'has_ai': 'AI'
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Avg Rank": st.column_config.NumberColumn(format="%.1f"),
                "B-T": st.column_config.NumberColumn(format="%d"),
                "AP": st.column_config.NumberColumn(format="%d"),
                "Coaches": st.column_config.NumberColumn(format="%d"),
                "Spread": st.column_config.NumberColumn(format="%d", help="Difference between highest and lowest rank"),
                "W": st.column_config.NumberColumn(format="%d"),
                "L": st.column_config.NumberColumn(format="%d"),
                "Win %": st.column_config.NumberColumn(format="%.3f"),
                "AI": st.column_config.CheckboxColumn("AI Analysis"),
            },
            height=600
        )
        
        st.caption("Spread = Difference between highest and lowest rank across all systems | NaN = Not ranked in that system")
        
        # Insights
        st.markdown("#### 🔍 Key Insights")
        col1, col2, col3 = st.columns(3)
        
        consensus_teams = len(df_combined[df_combined['num_polls_ranked'] == 3])
        col1.metric("Consensus Top 25", f"{consensus_teams} teams", help="Teams ranked in all three systems")
        
        # Most controversial team
        most_controversial = df_combined.nlargest(1, 'rank_spread').iloc[0] if len(df_combined) > 0 and df_combined['rank_spread'].max() > 0 else None
        if most_controversial is not None:
            col2.metric(
                "Most Controversial",
                most_controversial['team_name'],
                f"±{int(most_controversial['rank_spread'])} rank spread",
                help="Biggest difference between rankings across systems"
            )
        
        # Biggest overperformer in polls vs B-T
        if has_ap:
            df_combined['poll_overperform'] = df_combined.apply(
                lambda row: (row['bt_rank'] - row['ap_rank']) if pd.notna(row['bt_rank']) and pd.notna(row['ap_rank']) else 0,
                axis=1
            )
            biggest_overperform = df_combined.nlargest(1, 'poll_overperform').iloc[0] if df_combined['poll_overperform'].max() > 0 else None
            if biggest_overperform is not None:
                col3.metric(
                    "Poll Overperformer",
                    biggest_overperform['team_name'],
                    f"+{int(biggest_overperform['poll_overperform'])} spots",
                    help="Ranked much higher in polls than Bradley-Terry suggests"
                )
    
    st.markdown("---")

    # ========================================================================
    # 3-3) Individual Team Ranking History
    # ========================================================================
    st.markdown("### 📈 Individual Team Ranking History")
    
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

    if df_hist.empty:
        st.warning("The table bt.model_ranking_history is currently empty.")
    else:
        team_options = df_hist["team_name"].dropna().unique().tolist()
        
        # Default to the team selected in rank selector if available
        default_team = selected_team_name if selected_team_id is not None else team_options[0]
        default_index = team_options.index(default_team) if default_team in team_options else 0
        
        selected_team_history = st.selectbox(
            "Select a team to view ranking history", 
            sorted(team_options),
            index=default_index,
            key="history_team_select"
        )

        df_team = df_hist[df_hist["team_name"] == selected_team_history].sort_values("updated_at")

        c1, c2 = st.columns(2)

        # Rank over time
        rank_fig = px.line(
            df_team,
            x="updated_at",
            y="rank",
            title=f"{selected_team_history} – Rank over time (lower is better)",
            markers=True
        )
        rank_fig.update_yaxes(autorange="reversed")
        rank_fig.update_layout(hovermode='x unified')
        c1.plotly_chart(rank_fig, use_container_width=True)

        # Strength over time
        strength_fig = px.line(
            df_team,
            x="updated_at",
            y="strength",
            title=f"{selected_team_history} – Strength over time",
            markers=True
        )
        strength_fig.update_layout(hovermode='x unified')
        c2.plotly_chart(strength_fig, use_container_width=True)
        
        # Summary statistics
        if len(df_team) > 1:
            col1, col2, col3, col4 = st.columns(4)
            
            best_rank = df_team['rank'].min()
            worst_rank = df_team['rank'].max()
            current_rank = df_team.iloc[-1]['rank']
            rank_change = df_team.iloc[0]['rank'] - current_rank
            
            col1.metric("Current Rank", f"#{int(current_rank)}")
            col2.metric("Best Rank", f"#{int(best_rank)}")
            col3.metric("Worst Rank", f"#{int(worst_rank)}")
            col4.metric("Season Change", f"{int(rank_change):+d}", 
                       delta_color="normal" if rank_change > 0 else "inverse")

        st.markdown("---")

    # ========================================================================
    # 3-5) Understanding Bradley-Terry Rankings
    # ========================================================================
    with st.expander("📚 About Bradley-Terry Rankings"):
        st.markdown("""
        ### 🧮 What is Bradley-Terry Model?
        
        The **Bradley-Terry model** is a statistical method for ranking teams based on pairwise comparisons (head-to-head games).
        
        #### 🔍 How It Works:
        
        1. **Pairwise Comparisons**: Each game is a comparison between two teams
        2. **Probability-Based**: Models the probability that Team A beats Team B
        3. **Strength Parameter**: Each team has a "strength" value
        4. **Iterative Calculation**: Rankings emerge from solving a system of equations
        
        #### 📊 Key Components:
        
        **Strength (θ)**
        - Numerical value representing team's ability
        - Higher strength = better team
        - Typically ranges from ~0.5 to ~2.0
        - League average ≈ 1.0
        
        **Rank**
        - Ordinal position (1st, 2nd, 3rd, ...)
        - Based on strength values
        - Lower number = better
        
        **Win Probability vs Average**
        - Chance of beating an average team (strength = 1.0)
        - Example: 65% means favored against average opponent
        
        #### ✅ Advantages:
        
        - **Objective**: Based purely on game results
        - **Accounts for Schedule**: Beating strong teams matters more
        - **Transitive**: If A > B and B > C, then A > C (usually)
        - **No Human Bias**: No voting or subjective opinions
        
        #### ⚠️ Limitations:
        
        - **No Context**: Doesn't know about injuries, weather, momentum
        - **Assumes Consistency**: Doesn't account for teams improving/declining
        - **Sample Size**: Early season rankings less reliable
        - **Close Games**: Treats 1-point win same as blowout
        
        #### 🆚 vs Traditional Polls:
        
        | Aspect | Bradley-Terry | AP/Coaches Poll |
        |--------|---------------|-----------------|
        | **Method** | Statistical model | Human votes |
        | **Objectivity** | 100% objective | Subjective |
        | **Schedule Aware** | Yes, built-in | Voters may consider |
        | **Recency Bias** | No | Yes (recent games weighted more) |
        | **Eye Test** | No | Yes (quality of play matters) |
        | **Lag** | Updated frequently | Weekly updates |
        
        #### 💡 Best Use:
        
        - **For Analysis**: Most objective measure of team strength
        - **For Predictions**: Good for forecasting game outcomes
        - **For Comparison**: Pair with AP/Coaches to get full picture
        
        **Bottom Line:** Bradley-Terry gives you the "math says," while polls give you "experts say." 
        Truth often lies somewhere in between!
        """)

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


# # ============================================================================
# # 5. League Analytics
# # ============================================================================
# elif page == "5. League Analytics":
#     st.subheader("📚 League-wide Analytics – Distribution, Correlation, Clusters")

#     with st.spinner("Loading league stats..."):
#         df_stats, benchmark = load_team_stats_with_benchmark()

#     tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation", "Clustering"])

#     # -------------------------------
#     # Tab 1 – Distribution
#     # -------------------------------
#     with tab1:
#         st.markdown("### 📈 Metric Distribution")

#         metric_for_dist = st.selectbox(
#             "Choose a metric:",
#             TEAM_METRICS,
#             index=TEAM_METRICS.index("win_pct"),
#             format_func=metric_label,
#             key="league_dist_metric",
#         )

#         col1, col2 = st.columns(2)

#         fig_hist = px.histogram(
#             df_stats,
#             x=metric_for_dist,
#             nbins=20,
#             title=f"{metric_label(metric_for_dist)} – Histogram",
#         )
#         fig_hist.add_vline(
#             x=benchmark[metric_for_dist],
#             line_dash="dash",
#             annotation_text="Median",
#             annotation_position="top left",
#         )
#         col1.plotly_chart(fig_hist, use_container_width=True)

#         fig_box = px.box(
#             df_stats,
#             y=metric_for_dist,
#             title=f"{metric_label(metric_for_dist)} – Box Plot",
#         )
#         col2.plotly_chart(fig_box, use_container_width=True)

#     # -------------------------------
#     # Tab 2 – Correlation
#     # -------------------------------
#     with tab2:
#         st.markdown("### 🔗 Correlation Heatmap (Team-level metrics)")

#         corr_df = df_stats[TEAM_METRICS].corr()

#         fig_corr = px.imshow(
#             corr_df,
#             x=[metric_label(c) for c in TEAM_METRICS],
#             y=[metric_label(c) for c in TEAM_METRICS],
#             color_continuous_scale="RdBu",
#             origin="lower",
#             title="Correlation between Metrics",
#         )
#         st.plotly_chart(fig_corr, use_container_width=True)

#         st.markdown("### Scatter – Explore relationships")

#         x_metric = st.selectbox(
#             "X-axis metric:",
#             TEAM_METRICS,
#             index=TEAM_METRICS.index("avg_points_scored"),
#             format_func=metric_label,
#         )
#         y_metric = st.selectbox(
#             "Y-axis metric:",
#             TEAM_METRICS,
#             index=TEAM_METRICS.index("avg_points_allowed"),
#             format_func=metric_label,
#         )

#         fig_sc = px.scatter(
#             df_stats,
#             x=x_metric,
#             y=y_metric,
#             hover_name="team_name",
#             title=f"{metric_label(x_metric)} vs {metric_label(y_metric)}",
#         )
#         st.plotly_chart(fig_sc, use_container_width=True)

#     # -------------------------------
#     # Tab 3 – Clustering
#     # -------------------------------
#     with tab3:
#         st.markdown("### 🧬 Clustering – Group teams by style / performance")

#         cluster_metrics = st.multiselect(
#             "Choose 2–8 metrics for clustering:",
#             TEAM_METRICS,
#             default=[
#                 "win_pct",
#                 "avg_points_scored",
#                 "avg_points_allowed",
#                 "avg_total_yards",
#                 "avg_yards_allowed",
#             ],
#             format_func=metric_label,
#         )

#         if len(cluster_metrics) < 2:
#             st.info("Please select at least 2 metrics for clustering.")
#         else:
#             k = st.slider("Number of clusters (k)", min_value=2, max_value=6, value=3)

#             X = df_stats[cluster_metrics].fillna(df_stats[cluster_metrics].median())
#             X_std = (X - X.mean()) / X.std(ddof=0)

#             km = KMeans(n_clusters=k, random_state=42, n_init="auto")
#             labels = km.fit_predict(X_std)

#             df_cluster = df_stats.copy()
#             df_cluster["cluster"] = labels

#             st.markdown("#### Cluster assignment table")
#             st.dataframe(
#                 df_cluster[["team_name", "cluster"] + cluster_metrics],
#                 use_container_width=True,
#                 hide_index=True,
#             )

#             st.markdown("#### 2D projection for visualization")

#             x_metric = cluster_metrics[0]
#             y_metric = cluster_metrics[1]

#             fig_cluster = px.scatter(
#                 df_cluster,
#                 x=x_metric,
#                 y=y_metric,
#                 color="cluster",
#                 hover_name="team_name",
#                 title=f"Clusters by {metric_label(x_metric)} & {metric_label(y_metric)}",
#             )
#             st.plotly_chart(fig_cluster, use_container_width=True)


# ============================================================================
# 6. Text to SQL
# ============================================================================
elif page == "6. Text to SQL":
    st.subheader("💬 Text to SQL (MotherDuck)")
    st.caption("Ask a question in natural language. An LLM will propose SQL, then we run it on MotherDuck and show the results.")

    # Prompt examples + history selector
    sample_prompts = [
        "Top 25 teams by average score in 2025 season",
        "Show the AP Poll weekly ranking history for Oregon Ducks in the 2025 season",
        "Compare Ohio State Buckeyes and Michigan Wolverines: avg_points_scored, avg_points_allowed, win_pct",
        "Top 5 defenses by avg_points_allowed in 2025",
    ]
    with st.expander("Prompt examples", expanded=False):
        cols = st.columns(2)
        for i, p in enumerate(sample_prompts):
            if cols[i % 2].button(p, key=f"sample_{i}"):
                st.session_state["text2sql_question"] = p

    history = st.session_state.get("text2sql_history", [])
    def set_from_history():
        sel = st.session_state.get("text2sql_history_select")
        if sel:
            st.session_state["text2sql_question"] = sel

    st.selectbox(
        "Your previous questions",
        options=[""] + history,
        format_func=lambda x: "Select a past question" if x == "" else x,
        key="text2sql_history_select",
        on_change=set_from_history,
    )

    question = st.text_area(
        "Your question",
        value=st.session_state.get("text2sql_question", ""),
        placeholder="e.g., Top 5 teams by average points scored in 2025 season",
        key="text2sql_question",
    )

    run_btn = st.button("Generate and Run SQL", type="primary", disabled=not question.strip(), key="text2sql_run")

    # Cached results to keep chart controls usable after selection changes
    last_sql = st.session_state.get("text2sql_sql")
    last_df = st.session_state.get("text2sql_df")

    if run_btn:
        with st.spinner("Generating SQL via LLM..."):
            sql = generate_sql_from_text(question.strip())

        if sql:
            st.markdown("#### Generated SQL")
            st.code(sql, language="sql")

            if not is_sql_safe(sql):
                st.error("Generated SQL contains write/DDL operations. Blocking execution.")
                st.stop()

            with st.spinner("Running query..."):
                try:
                    df = run_query(sql)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Query failed: {e}")
                    df = None

            if df is not None:
                st.session_state["text2sql_sql"] = sql
                st.session_state["text2sql_df"] = df
                # keep a short history of unique questions
                if question.strip():
                    hist = st.session_state.get("text2sql_history", [])
                    if question.strip() not in hist:
                        hist = [question.strip()] + hist
                        st.session_state["text2sql_history"] = hist[:10]
                if df.empty:
                    st.info("Query ran successfully but returned no rows.")
                else:
                    st.success(f"Returned {len(df)} rows.")
                    df_show = df.copy()
                    df_show.insert(0, "#", range(1, len(df_show) + 1))
                    st.dataframe(df_show, use_container_width=True)

                    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    if numeric_cols:
                        st.markdown("#### Quick Chart")
                        non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
                        default_x = (
                            "week_number"
                            if "week_number" in df.columns
                            else (non_numeric_cols[0] if non_numeric_cols else df.columns[0])
                        )
                        default_y = "current_rank" if "current_rank" in numeric_cols else numeric_cols[0]
                        x_col = st.selectbox("X axis", df.columns, index=list(df.columns).index(default_x), key="text2sql_x")
                        y_col = st.selectbox("Y axis (numeric)", numeric_cols, index=numeric_cols.index(default_y), key="text2sql_y")
                        chart_type = st.selectbox("Chart type", ["Bar", "Line", "Scatter"], key="text2sql_chart")

                        try:
                            df_plot = df.copy()
                            if df_plot[x_col].dtype.kind in "if" and "week" in x_col.lower():
                                df_plot[x_col] = df_plot[x_col].round().astype(int)

                            if chart_type == "Bar":
                                fig = px.bar(df_plot, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                            elif chart_type == "Line":
                                fig = px.line(df_plot, x=x_col, y=y_col, title=f"{y_col} over {x_col}", markers=True)
                            else:
                                fig = px.scatter(df_plot, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:  # noqa: BLE001
                            st.warning(f"Unable to render chart: {e}")

    # Re-display cached results so chart controls don't reset on selection changes
    if not run_btn and last_df is not None and not last_df.empty:
        st.markdown("#### Generated SQL (cached)")
        st.code(last_sql or "", language="sql")
        st.success(f"Returned {len(last_df)} rows.")
        df_show = last_df.copy()
        df_show.insert(0, "#", range(1, len(df_show) + 1))
        st.dataframe(df_show, use_container_width=True)

        numeric_cols = last_df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            st.markdown("#### Quick Chart")
            non_numeric_cols = last_df.select_dtypes(exclude=["number"]).columns.tolist()
            default_x = (
                "week_number"
                if "week_number" in last_df.columns
                else (non_numeric_cols[0] if non_numeric_cols else last_df.columns[0])
            )
            default_y = "current_rank" if "current_rank" in numeric_cols else numeric_cols[0]

            x_val = st.session_state.get("text2sql_x", default_x)
            if x_val not in last_df.columns:
                x_val = default_x
            y_val = st.session_state.get("text2sql_y", default_y)
            if y_val not in numeric_cols:
                y_val = default_y
            chart_val = st.session_state.get("text2sql_chart", "Bar")

            x_col = st.selectbox("X axis", last_df.columns, index=list(last_df.columns).index(x_val), key="text2sql_x")
            y_col = st.selectbox("Y axis (numeric)", numeric_cols, index=numeric_cols.index(y_val), key="text2sql_y")
            chart_type = st.selectbox("Chart type", ["Bar", "Line", "Scatter"], index=["Bar","Line","Scatter"].index(chart_val) if chart_val in ["Bar","Line","Scatter"] else 0, key="text2sql_chart")

            try:
                df_plot = last_df.copy()
                if df_plot[x_col].dtype.kind in "if" and "week" in x_col.lower():
                    df_plot[x_col] = df_plot[x_col].round().astype(int)

                if chart_type == "Bar":
                    fig = px.bar(df_plot, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                elif chart_type == "Line":
                    fig = px.line(df_plot, x=x_col, y=y_col, title=f"{y_col} over {x_col}", markers=True)
                else:
                    fig = px.scatter(df_plot, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:  # noqa: BLE001
                st.warning(f"Unable to render chart: {e}")
