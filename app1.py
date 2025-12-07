import os
from typing import List, Dict

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    # DuckDB uses $1, $2, etc. for parameter placeholders, not ?
    # We need to replace ? with $1, $2, etc.
    param_count = sql.count('?')
    for i in range(param_count):
        sql = sql.replace('?', f'${i+1}', 1)
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
        "5. League Analytics",
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
        
    st.info("💡 **How to read:** Values > 1.0 (green) indicate above-median performance. Values < 1.0 (red) indicate below-median performance.")

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
