import os
import duckdb
import pandas as pd
import streamlit as st

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
    使用環境變數 MOTHERDUCK_TOKEN 連到 MotherDuck 上的 `ncaa` database。
    """
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        st.error(
            "環境變數 `MOTHERDUCK_TOKEN` 尚未設定。\n\n"
            "請先在終端機 / Cloud Shell 執行：\n"
            "    export MOTHERDUCK_TOKEN='md:你的token'\n\n"
            "然後再重新啟動 Streamlit。"
        )
        st.stop()

    md_db_name = "ncaa"
    conn = duckdb.connect(f"md:{md_db_name}")
    return conn


def run_query(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """小工具：執行 SQL 並回傳 DataFrame。"""
    conn = get_connection()
    if params is None:
        return conn.execute(sql).fetch_df()
    return conn.execute(sql, params).fetch_df()

# ------------------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------------------
st.set_page_config(page_title="BA882 College Football Dashboard", layout="wide")
st.title("College Football Analytics – BA882 Team 2")
st.caption("Data source: ESPN Hidden API → GCS → MotherDuck (via Airflow)")


# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
page = st.sidebar.radio(
    "選擇報表頁面：",
    [
        "Overview",
        "Team Rankings (BT)",
        "Team Stats (Aggregates)",
        "Game Explorer",
        "Pairwise Matchups (BT Inputs)",
        "Ranking History (BT)",
    ],
)

# ------------------------------------------------------------------------------
# 頁面 1：Overview
# ------------------------------------------------------------------------------
if page == "Overview":
    st.subheader("📊 Project Overview")

    # 1) 基本統計
    col1, col2, col3 = st.columns(3)

    df_teams = run_query("SELECT COUNT(*) AS n_teams FROM real_deal.dim_teams;")
    df_games = run_query("SELECT COUNT(*) AS n_games FROM real_deal.dim_games;")
    df_season = run_query(
        "SELECT MIN(season) AS min_season, MAX(season) AS max_season "
        "FROM real_deal.dim_games;"
    )

    col1.metric("Number of Teams", int(df_teams["n_teams"].iloc[0]))
    col2.metric("Number of Games", int(df_games["n_games"].iloc[0]))
    col3.metric(
        "Season Range",
        f"{int(df_season['min_season'].iloc[0])} – {int(df_season['max_season'].iloc[0])}",
    )

    st.markdown("---")

    # 2) Poll rankings coverage
    st.markdown("### 🏆 Ranking Poll Coverage")
    df_polls = run_query(
        """
        SELECT
            poll_name,
            MIN(season_year) AS min_season,
            MAX(season_year) AS max_season,
            COUNT(*) AS n_rows
        FROM real_deal.fact_rankings
        GROUP BY 1
        ORDER BY 1;
        """
    )
    st.dataframe(df_polls, use_container_width=True)

    # ------------------------------------------------------------------------------
# 頁面 2：Team Rankings (BT)
# ------------------------------------------------------------------------------
elif page == "Team Rankings (BT)":
    st.subheader("🏅 Bradley–Terry Team Rankings")

    # Filter: Top N, team name 搜尋
    col_left, col_right = st.columns([1, 2])
    top_n = col_left.slider("顯示前 Top N 球隊", min_value=10, max_value=150, value=50, step=10)
    name_filter = col_right.text_input("搜尋球隊名稱 (包含字串)", "")

    # 讀取 ranking + team name
    sql_rank = """
        SELECT
            r.rank,
            r.team_id,
            t.display_name AS team_name,
            r.strength,
            r.prob_vs_avg,
            r.updated_at
        FROM bt.rankings AS r
        LEFT JOIN real_deal.dim_teams AS t
            ON r.team_id = t.id
        ORDER BY r.rank
        LIMIT ?;
    """
    df_rank = run_query(sql_rank, (top_n,))

    if name_filter:
        mask = df_rank["team_name"].str.contains(name_filter, case=False, na=False)
        df_rank = df_rank[mask]

    # 顯示表格
    st.markdown("#### Current BT Rankings")
    st.dataframe(
        df_rank,
        use_container_width=True,
        hide_index=True,
    )

    # 簡單圖：Rank vs Strength
    st.markdown("#### Strength vs Rank")
    if not df_rank.empty:
        chart_df = df_rank[["rank", "strength", "team_name"]].set_index("rank")
        st.line_chart(chart_df[["strength"]])


    # ------------------------------------------------------------------------------
# 頁面 3：Team Stats (Aggregates)
# ------------------------------------------------------------------------------
elif page == "Team Stats (Aggregates)":
    st.subheader("📈 Team Season Aggregates (bt.team_stats + dim_teams)")

    # 先取 team list
    df_team_list = run_query(
        """
        SELECT
            ts.team_id,
            COALESCE(t.display_name, t.name) AS team_name
        FROM bt.team_stats AS ts
        LEFT JOIN real_deal.dim_teams AS t
            ON ts.team_id = t.id
        ORDER BY team_name;
        """
    )

    team_options = df_team_list["team_name"].tolist()
    team_map = dict(zip(df_team_list["team_name"], df_team_list["team_id"]))

    selected_team_name = st.selectbox("選擇球隊", team_options)
    selected_team_id = team_map[selected_team_name]

    # 讀取該隊的 stats
    sql_stats = """
        SELECT
            ts.*,
            COALESCE(t.display_name, t.name) AS team_name
        FROM bt.team_stats AS ts
        LEFT JOIN real_deal.dim_teams AS t
            ON ts.team_id = t.id
        WHERE ts.team_id = ?;
    """
    df_stats = run_query(sql_stats, (selected_team_id,))

    if df_stats.empty:
        st.warning("這支球隊暫時沒有 team_stats 資料。")
    else:
        row = df_stats.iloc[0]

        # 上面：關鍵指標 cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Games Played", int(row["games_played"]))
        c2.metric("Record (W-L-T)", f"{int(row['wins'])}-{int(row['losses'])}-{int(row['ties'])}")
        c3.metric("Win %", f"{row['win_pct']:.3f}")
        c4.metric("Point Differential", f"{row['point_differential']:.1f}")

        st.markdown("---")

        # 左右 layout：Offense vs Defense 摘要 + raw table
        col_off, col_def = st.columns(2)

        # Offense 指標
        off_metrics = {
            "Avg Points Scored": row["avg_points_scored"],
            "Avg Total Yards": row["avg_total_yards"],
            "Yards per Pass": row["avg_yards_per_pass"],
            "Yards per Rush": row["avg_yards_per_rush"],
            "Points per Yard (Offense)": row["points_per_yard_offense"],
        }
        col_off.markdown("##### 🔥 Offense Summary")
        off_df = pd.DataFrame(
            {"metric": list(off_metrics.keys()), "value": list(off_metrics.values())}
        )
        col_off.dataframe(off_df, hide_index=True, use_container_width=True)

        # Defense 指標
        def_metrics = {
            "Avg Points Allowed": row["avg_points_allowed"],
            "Avg Yards Allowed": row["avg_yards_allowed"],
            "Opp 3rd Eff": row["avg_third_eff_allowed"],
            "Opp 4th Eff": row["avg_fourth_eff_allowed"],
            "Points per Yard (Defense)": row["points_per_yard_defense"],
        }
        col_def.markdown("##### 🛡 Defense Summary")
        def_df = pd.DataFrame(
            {"metric": list(def_metrics.keys()), "value": list(def_metrics.values())}
        )
        col_def.dataframe(def_df, hide_index=True, use_container_width=True)

        st.markdown("#### Raw bt.team_stats row")
        st.dataframe(df_stats, use_container_width=True)


# ------------------------------------------------------------------------------
# 頁面 4：Game Explorer
# ------------------------------------------------------------------------------
elif page == "Game Explorer":
    st.subheader("🧭 Game Explorer (real_deal.fact_game_team + dim_games + dim_teams)")

    # 先抓 season, week 範圍
    df_season_week = run_query(
        """
        SELECT DISTINCT season, week
        FROM real_deal.dim_games
        ORDER BY season DESC, week ASC;
        """
    )

    seasons = sorted(df_season_week["season"].unique(), reverse=True)
    selected_season = st.selectbox("Season", seasons)

    weeks_in_season = df_season_week[df_season_week["season"] == selected_season]["week"].tolist()
    selected_week = st.selectbox("Week", weeks_in_season)

    st.markdown(f"顯示 {selected_season} Season, Week {selected_week} 的所有比賽。")

    # 查詢每一場比賽 (Home vs Away)
    sql_games = """
        SELECT
            g.id AS game_id,
            g.start_date,
            g.season,
            g.week,
            home_team.display_name AS home_team,
            away_team.display_name AS away_team,
            home_ft.score AS home_score,
            away_ft.score AS away_score,
            v.fullname AS venue
        FROM real_deal.dim_games AS g
        JOIN real_deal.fact_game_team AS home_ft
            ON g.id = home_ft.game_id AND home_ft.home_away = 'home'
        JOIN real_deal.fact_game_team AS away_ft
            ON g.id = away_ft.game_id AND away_ft.home_away = 'away'
        JOIN real_deal.dim_teams AS home_team
            ON home_ft.team_id = home_team.id
        JOIN real_deal.dim_teams AS away_team
            ON away_ft.team_id = away_team.id
        LEFT JOIN real_deal.dim_venues AS v
            ON g.venue_id = v.id
        WHERE g.season = ? AND g.week = ?
        ORDER BY g.start_date, home_team.display_name;
    """
    df_games = run_query(sql_games, (int(selected_season), int(selected_week)))

    st.dataframe(df_games, use_container_width=True, hide_index=True)

    # 簡單圖：比分差
    if not df_games.empty:
        st.markdown("#### Score Margin (Home - Away)")
        plot_df = df_games.copy()
        plot_df["score_margin"] = plot_df["home_score"] - plot_df["away_score"]
        plot_df = plot_df.set_index("game_id")[["score_margin"]]
        st.bar_chart(plot_df)

# ------------------------------------------------------------------------------
# 頁面 5：Pairwise Matchups (BT Inputs)
# ------------------------------------------------------------------------------
elif page == "Pairwise Matchups (BT Inputs)":
    st.subheader("⚖️ Pairwise Matchups – BT Model Inputs")

    # 用 dim_games 拿 season / week
    df_season_week = run_query(
        """
        SELECT DISTINCT season, week
        FROM real_deal.dim_games
        ORDER BY season DESC, week ASC;
        """
    )

    if df_season_week.empty:
        st.warning("real_deal.dim_games 沒有資料。")
    else:
        seasons = sorted(df_season_week["season"].unique(), reverse=True)
        selected_season = st.selectbox("Season", seasons, key="pw_season")

        weeks_in_season = (
            df_season_week[df_season_week["season"] == selected_season]["week"]
            .dropna()
            .tolist()
        )
        selected_week = st.selectbox("Week", weeks_in_season, key="pw_week")

        st.markdown(
            f"顯示 **{selected_season} Season, Week {selected_week}** 的 pairwise comparison 資料。"
        )

        sql_pw = """
            SELECT
                g.id AS game_id,
                g.start_date,
                g.season,
                g.week,
                home_team.display_name AS home_team,
                away_team.display_name AS away_team,
                pc.home_score,
                pc.away_score,
                pc.score_margin,
                pc.home_total_yards,
                pc.away_total_yards,
                (pc.home_total_yards - pc.away_total_yards) AS yard_margin,
                pc.home_third_eff,
                pc.away_third_eff,
                pc.home_fourth_eff,
                pc.away_fourth_eff,
                pc.home_yards_per_pass,
                pc.away_yards_per_pass,
                pc.home_yards_per_rush,
                pc.away_yards_per_rush,
                pc.home_turnovers,
                pc.away_turnovers
            FROM bt.pairwise_comparisons AS pc
            JOIN real_deal.dim_games AS g
                ON pc.game_id = g.id
            JOIN real_deal.dim_teams AS home_team
                ON pc.home_team_id = home_team.id
            JOIN real_deal.dim_teams AS away_team
                ON pc.away_team_id = away_team.id
            WHERE g.season = ? AND g.week = ?
            ORDER BY g.start_date, home_team.display_name;
        """
        df_pw = run_query(sql_pw, (int(selected_season), int(selected_week)))
        st.markdown("#### Game-level Pairwise Features")
        st.dataframe(df_pw, use_container_width=True, hide_index=True)

        if not df_pw.empty:
            # Home win rate
            home_win_rate = (df_pw["score_margin"] > 0).mean()
            st.metric("Home Win Rate (this week)", f"{home_win_rate:.1%}")

            st.markdown("#### Yard Margin vs Score Margin")
            scatter_df = df_pw.copy()
            scatter_df["yard_margin"] = scatter_df["yard_margin"].astype(float)
            scatter_df["score_margin"] = scatter_df["score_margin"].astype(float)

            st.scatter_chart(
                scatter_df,
                x="yard_margin",
                y="score_margin",
            )


# ------------------------------------------------------------------------------
# 頁面 6：Ranking History (BT)
# ------------------------------------------------------------------------------
elif page == "Ranking History (BT)":
    st.subheader("📉 Ranking History – BT Model")

    # 先抓有哪些隊出現在歷史 ranking 中
    df_team_hist = run_query(
        """
        SELECT DISTINCT
            h.team_id,
            COALESCE(t.display_name, t.name) AS team_name
        FROM bt.model_ranking_history AS h
        LEFT JOIN real_deal.dim_teams AS t
            ON h.team_id = t.id
        ORDER BY team_name;
        """
    )

    if df_team_hist.empty:
        st.warning("bt.model_ranking_history 目前沒有資料。")
    else:
        team_options = df_team_hist["team_name"].tolist()
        team_map = dict(zip(df_team_hist["team_name"], df_team_hist["team_id"]))

        selected_team_name = st.selectbox("選擇球隊", team_options, key="hist_team")
        selected_team_id = team_map[selected_team_name]

        sql_hist = """
            SELECT
                h.updated_at,
                h.rank,
                h.strength,
                h.prob_vs_avg
            FROM bt.model_ranking_history AS h
            WHERE h.team_id = ?
            ORDER BY h.updated_at;
        """
        df_hist = run_query(sql_hist, (selected_team_id,))

        if df_hist.empty:
            st.warning("這支球隊目前沒有 ranking history。")
        else:
            st.markdown(f"#### {selected_team_name} – Rank & Strength Over Time")

            # Strength over time
            st.markdown("**Strength over time**")
            str_df = df_hist[["updated_at", "strength"]].set_index("updated_at")
            st.line_chart(str_df)

            # Rank over time（名次越小越好）
            st.markdown("**Rank over time**（數字越小越好）")
            rank_df = df_hist[["updated_at", "rank"]].set_index("updated_at")
            st.line_chart(rank_df)

            st.markdown("#### Raw ranking history data")
            st.dataframe(df_hist, use_container_width=True, hide_index=True)