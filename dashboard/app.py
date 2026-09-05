"""Cricket analytics dashboard.

    streamlit run dashboard/app.py

Reads the Hive tables produced by the pipeline, or CSV snapshots when Hive is
unreachable. The active source is always shown in the sidebar.
"""
import altair as alt
import pandas as pd
import streamlit as st

import data_access as da

# Categorical slots 1-3 of the reference palette, in fixed order. Validated
# all-pairs in both modes; do not cycle or extend without re-validating.
PHASE_COLORS = {"powerplay": "#2a78d6", "middle": "#eb6834", "death": "#1baf7a"}
PHASE_ORDER = ["powerplay", "middle", "death"]
LINE_COLOR = "#2a78d6"
GRID_COLOR = "#e4e3df"
TEXT_SECONDARY = "#52514e"

st.set_page_config(page_title="Cricket Analytics", layout="wide")


@alt.theme.register("cricket", enable=True)
def chart_theme():
    return alt.theme.ThemeConfig({
        "config": {
            "view": {"strokeWidth": 0},
            "axis": {"grid": True, "gridColor": GRID_COLOR, "gridOpacity": 0.7,
                     "domainColor": GRID_COLOR, "tickColor": GRID_COLOR,
                     "labelColor": TEXT_SECONDARY, "titleColor": TEXT_SECONDARY,
                     "labelFontSize": 11, "titleFontSize": 11},
            "legend": {"labelColor": TEXT_SECONDARY, "titleColor": TEXT_SECONDARY},
        }
    })


@st.cache_resource
def get_connection():
    return da.connect()


conn, fallback_reason = get_connection()

st.sidebar.title("Cricket Analytics")
st.sidebar.caption(f"Data source: **{conn.label}**")
if fallback_reason:
    st.sidebar.warning(
        f"Hive unavailable, serving CSV snapshots.\n\n`{fallback_reason[:200]}`")

view = st.sidebar.radio(
    "View", ["Bowler economy by phase", "Match explorer", "What-if win probability"])


# --------------------------------------------------------------- view 1
if view == "Bowler economy by phase":
    st.header("Bowler economy by phase")
    st.caption("Output of the Hadoop Streaming MapReduce job. Byes and legbyes "
               "are not charged to the bowler; wides and no-balls are.")

    c1, c2, c3 = st.columns(3)
    match_type = c1.selectbox("Match type", ["T20", "IT20", "ODI", "ODM"])
    phase = c2.selectbox("Phase", PHASE_ORDER + ["(all)"])
    min_balls = c3.slider("Minimum legal balls bowled", 60, 3000, 600, step=60)

    try:
        df = da.bowler_economy(conn, match_type,
                               None if phase == "(all)" else phase, min_balls)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Query failed: {exc}")
        st.stop()

    if df.empty:
        st.info("No bowlers match those filters.")
        st.stop()

    top = df.nsmallest(20, "economy")
    bars = (alt.Chart(top)
            .mark_bar(cornerRadiusEnd=4, height=14)
            .encode(
                x=alt.X("economy:Q", title="Economy (runs per over)"),
                y=alt.Y("bowler_id:N", sort="x", title="Bowler id"),
                color=alt.Color("phase:N",
                                scale=alt.Scale(domain=PHASE_ORDER,
                                                range=[PHASE_COLORS[p] for p in PHASE_ORDER]),
                                legend=alt.Legend(title="Phase")),
                tooltip=["bowler_id", "phase", "match_type",
                         alt.Tooltip("economy:Q", format=".2f"),
                         alt.Tooltip("overs_bowled:Q", format=".1f"),
                         "wickets"])
            .properties(height=28 * len(top), title="Most economical bowlers"))
    labels = (alt.Chart(top).mark_text(align="left", dx=4, color=TEXT_SECONDARY,
                                       fontSize=11)
              .encode(x="economy:Q", y=alt.Y("bowler_id:N", sort="x"),
                      text=alt.Text("economy:Q", format=".2f")))
    st.altair_chart(bars + labels, width="stretch")

    with st.expander("Table view"):
        st.dataframe(df, width="stretch")


# --------------------------------------------------------------- view 2
elif view == "Match explorer":
    st.header("Match explorer")
    st.caption("Win probability is the model's estimate for the chasing team, "
               "recomputed after every ball of the second innings.")

    try:
        matches = da.available_matches(conn)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not list matches: {exc}")
        st.stop()

    if matches.empty:
        st.info("No scored matches yet - run ml/score_all_matches.py.")
        st.stop()

    def describe(row):
        parts = [str(row["match_id"])]
        if "team1" in row and pd.notna(row.get("team1")):
            parts.append(f"{row['team1']} v {row['team2']}")
        if "match_date" in row and pd.notna(row.get("match_date")):
            parts.append(str(row["match_date"]))
        return " - ".join(parts)

    options = {describe(r): r["match_id"] for _, r in matches.iterrows()}
    chosen = st.selectbox("Match", list(options))
    curve = da.win_probability_curve(conn, options[chosen])

    if curve.empty:
        st.info("No second-innings data for that match.")
        st.stop()

    hover = alt.selection_point(fields=["ball_seq"], nearest=True,
                                on="mouseover", empty=False)

    base = alt.Chart(curve).encode(
        x=alt.X("ball_seq:Q", title="Ball of the chase"))

    wp_line = base.mark_line(color=LINE_COLOR, strokeWidth=2).encode(
        y=alt.Y("win_probability:Q", title="Win probability",
                scale=alt.Scale(domain=[0, 1])))
    wp_points = base.mark_point(size=60, opacity=0).encode(
        y="win_probability:Q",
        tooltip=[alt.Tooltip("ball_seq:Q", title="Ball"),
                 alt.Tooltip("current_score:Q", title="Score"),
                 alt.Tooltip("wickets_down:Q", title="Wickets"),
                 alt.Tooltip("runs_required:Q", title="Runs needed"),
                 alt.Tooltip("required_run_rate:Q", title="Req. rate", format=".2f"),
                 alt.Tooltip("win_probability:Q", title="Win prob", format=".3f")]
    ).add_params(hover)
    midline = (alt.Chart(pd.DataFrame({"y": [0.5]}))
               .mark_rule(color=GRID_COLOR, strokeDash=[4, 4]).encode(y="y:Q"))

    st.altair_chart((midline + wp_line + wp_points).properties(
        height=280, title="Win probability through the chase"),
        width="stretch")

    # Runs and win probability live on different scales, so they get their own
    # charts rather than a second y-axis.
    score_line = base.mark_line(color=LINE_COLOR, strokeWidth=2).encode(
        y=alt.Y("current_score:Q", title="Runs"),
        tooltip=["ball_seq", "current_score", "wickets_down"])
    st.altair_chart(score_line.properties(height=200, title="Score progression"),
                    width="stretch")

    with st.expander("Ball-by-ball table"):
        st.dataframe(curve, width="stretch")


# --------------------------------------------------------------- view 3
else:
    st.header("What-if win probability")
    st.caption("Scores a hypothetical chase state against the trained model. "
               "Needs a local Spark session and the saved model.")

    c1, c2 = st.columns(2)
    match_type = c1.selectbox("Match type", ["T20", "IT20", "ODI", "ODM"])
    overs_cap = 20 if match_type in ("T20", "IT20") else 50
    target = c1.number_input("Target", min_value=1, value=180)
    score = c1.number_input("Current score", min_value=0, value=120)
    wickets = c2.slider("Wickets down", 0, 9, 4)
    balls_remaining = c2.slider("Balls remaining", 1, overs_cap * 6, 36)
    model_path = c2.text_input("Model path", "hdfs:///data/models/win_probability_gbt")

    if st.button("Estimate"):
        with st.spinner("Starting Spark and loading the model..."):
            try:
                import sys
                sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "ml"))
                from pyspark.ml import PipelineModel
                from pyspark.sql import SparkSession
                import predict_live_state as pls

                spark = SparkSession.builder.appName("whatif").getOrCreate()
                model = PipelineModel.load(model_path)
                row = pls.state_row(score, wickets, balls_remaining, target,
                                    match_type, overs_cap)
                prob = pls.predict(spark, model, row)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Scoring failed: {exc}")
            else:
                st.metric("Chasing team win probability", f"{prob:.1%}")
                st.caption(
                    f"Needs {max(target - score, 0)} runs from {balls_remaining} balls "
                    f"with {10 - wickets} wickets in hand.")
