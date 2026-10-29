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
    "View", ["Bowler economy by phase", "Match explorer", "Batter vs bowler",
             "Model scorecard", "What-if win probability"])
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order


# --------------------------------------------------------------- view 1
if view == "Bowler economy by phase":
    st.header("Bowler economy by phase")
    st.caption("Output of the Hadoop Streaming MapReduce job. Byes and legbyes "
               "are not charged to the bowler; wides and no-balls are. A dot ball is a "
               "legal ball on which the bowler is charged no runs.")

    st.info("These rankings pool men's, women's and associate-nation matches and make no "
            "adjustment for opposition. The very top places are usually bowlers from a strong "
            "side bowling against weak ones (for example, a T20 powerplay economy near 3.6 "
            "against a league average of 7.2), so read them as raw records, not as quality.")
    metric = st.radio("Rank bowlers by", ["Economy", "Dot-ball %"], horizontal=True)
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

    if metric == "Economy":
        col, axis_title, fmt = "economy", "Economy (runs per over)", ".2f"
        top = df.nsmallest(20, "economy")
    else:
        col, axis_title, fmt = "dot_ball_pct", "Dot balls (% of legal balls)", ".1f"
        top = df.dropna(subset=["dot_ball_pct"]).nlargest(20, "dot_ball_pct")
    order = top["bowler_id"].tolist()  # explicit: best first
    bars = (alt.Chart(top)
            .mark_bar(cornerRadiusEnd=4, height=14)
            .encode(
                x=alt.X(f"{col}:Q", title=axis_title),
                y=alt.Y("bowler_id:N", sort=order, title="Bowler id"),
                color=alt.Color("phase:N",
                                scale=alt.Scale(domain=PHASE_ORDER,
                                                range=[PHASE_COLORS[p] for p in PHASE_ORDER]),
                                legend=alt.Legend(title="Phase")),
                tooltip=["bowler_id", "phase", "match_type",
                         alt.Tooltip("economy:Q", format=".2f"),
                         alt.Tooltip("dot_ball_pct:Q", title="Dot-ball %", format=".1f"),
                         alt.Tooltip("overs_bowled:Q", format=".1f"),
                         "wickets"])
            .properties(height=28 * len(top),
                        title="Most economical bowlers" if metric == "Economy"
                        else "Bowlers who bowl the most dot balls"))
    labels = (alt.Chart(top).mark_text(align="left", dx=4, color=TEXT_SECONDARY,
                                       fontSize=11)
              .encode(x=f"{col}:Q", y=alt.Y("bowler_id:N", sort=order),
                      text=alt.Text(f"{col}:Q", format=fmt)))
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

    # Win probability added: the change in the chasing team's win probability caused by
    # one ball. Positive helps the chasing team; the first ball has no previous state.
    swing = curve.dropna(subset=["wpa"])
    if not swing.empty:
        wpa_bars = (alt.Chart(swing).mark_bar()
                    .encode(x=alt.X("ball_seq:Q", title="Ball of the chase"),
                            y=alt.Y("wpa:Q", title="Win probability added"),
                            color=alt.condition(alt.datum.wpa >= 0,
                                                alt.value("#1baf7a"), alt.value("#eb6834")),
                            tooltip=[alt.Tooltip("ball_seq:Q", title="Ball"), "batter", "bowler",
                                     alt.Tooltip("wpa:Q", format="+.3f", title="WPA")])
                    .properties(height=200, title="Win probability added per ball "
                                "(green helps the chasing team, orange hurts)"))
        st.altair_chart(wpa_bars, width="stretch")

        biggest = swing.reindex(swing["wpa"].abs().sort_values(ascending=False).index).head(5)
        st.subheader("Five biggest swings")
        st.dataframe(biggest[["ball_seq", "batter", "bowler", "prev_win_probability",
                              "win_probability", "wpa"]]
                     .rename(columns={"prev_win_probability": "before", "win_probability": "after"}),
                     hide_index=True, width="stretch")

    with st.expander("Ball-by-ball table"):
        st.dataframe(curve, width="stretch")


# --------------------------------------------------------------- view 3
elif view == "Batter vs bowler":
    st.header("Batter vs bowler")
    st.caption("A lookup of how often each batter has faced each bowler, not a predictor: "
               "most pairs have met for only a handful of balls. Pairs with fewer than 12 "
               "balls are not listed. 'Expected SR' is what the batter would be expected to "
               "score off this bowler from their own and the bowler's overall records; "
               "'adjusted SR' pulls small samples toward it.")

    fam = st.radio("Format", ["T20", "ODI", "ALL"], horizontal=True)
    try:
        pairs = da.matchup_lookup(conn, fam)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load matchups: {exc}")
        st.stop()

    side = st.radio("Look up by", ["Batter", "Bowler"], horizontal=True)
    own, other = ("batter", "bowler") if side == "Batter" else ("bowler", "batter")
    totals = pairs.groupby(f"{own}_name")["balls"].sum().sort_values(ascending=False)
    name = st.selectbox(f"{side} (most balls first)", list(totals.index))
    shown = pairs[pairs[f"{own}_name"] == name].sort_values("balls", ascending=False)

    tier_help = {"solid": "60+ balls", "moderate": "30-59 balls", "limited": "12-29 balls"}
    st.write(f"**{name}**: {len(shown)} {other}s with at least 12 balls "
             f"({', '.join(f'{(shown.tier == k).sum()} {k}' for k in tier_help)}).")
    table = shown[[f"{other}_name", "balls", "runs", "dismissals", "strike_rate", "dot_pct",
                   "expected_sr", "shrunk_sr", "z_vs_expected", "tier"]].rename(columns={
        f"{other}_name": other.capitalize(), "strike_rate": "SR", "dot_pct": "Dot %",
        "expected_sr": "Expected SR", "shrunk_sr": "Adjusted SR",
        "z_vs_expected": "z vs expected"})
    st.dataframe(table.round(2), hide_index=True, width="stretch")
    st.caption("z vs expected: how many standard errors the batter's strike rate is from "
               "expectation; beyond about +/-2 is noteworthy only for 'solid' pairs. "
               "Tiers: " + "; ".join(f"{k} = {v}" for k, v in tier_help.items()) + ".")


# --------------------------------------------------------------- view 4
elif view == "Model scorecard":
    st.header("Model scorecard")
    st.caption("How well the win-probability models do on seasons they never saw "
               "(trained before 2021/22, tested on 2021/22 onward).")

    try:
        comp = da.model_comparison(conn).sort_values("logloss")
        buckets = da.calibration_buckets(conn)
        imp = da.feature_importance(conn, "permutation")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load evaluation tables: {exc}")
        st.stop()

    st.subheader("Model comparison")
    st.write("Lower Log-Loss, Brier and ECE are better; higher AUC is better. "
             "Logistic Regression is as good as or better than the tree models.")
    st.dataframe(comp[["model", "auc", "brier", "logloss", "ece", "accuracy", "f1_win"]]
                 .rename(columns={"model": "Model", "auc": "AUC", "brier": "Brier",
                                  "logloss": "Log-Loss", "ece": "ECE", "accuracy": "Accuracy",
                                  "f1_win": "F1 (win)"}).round(4),
                 hide_index=True, width="stretch")

    st.subheader("Are the probabilities honest?")
    st.write("When the model says 70%, does the chasing team win about 70% of the time? "
             "Points on the diagonal mean yes. The raw model is over-optimistic on the "
             "held-out seasons; the calibrated version corrects most of it.")
    long = buckets.melt(id_vars=["bucket", "n", "actual"],
                        value_vars=["raw_pred", "calibrated_pred"],
                        var_name="version", value_name="predicted")
    long["version"] = long["version"].map({"raw_pred": "Raw model",
                                           "calibrated_pred": "Calibrated"})
    diag = (alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(color=GRID_COLOR, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q"))
    pts = (alt.Chart(long).mark_line(point=True, strokeWidth=2)
           .encode(x=alt.X("predicted:Q", title="Average predicted probability",
                           scale=alt.Scale(domain=[0, 1])),
                   y=alt.Y("actual:Q", title="Actual win rate", scale=alt.Scale(domain=[0, 1])),
                   color=alt.Color("version:N", scale=alt.Scale(
                       domain=["Raw model", "Calibrated"], range=MODEL_COLORS[:2]),
                       legend=alt.Legend(title=None)),
                   tooltip=["version", alt.Tooltip("predicted:Q", format=".3f"),
                            alt.Tooltip("actual:Q", format=".3f"), "n"]))
    st.altair_chart((diag + pts).properties(height=340, title="Reliability on held-out seasons"),
                    width="stretch")

    st.subheader("What the model relies on")
    st.write("Permutation importance: how much worse Log-Loss gets when one group of "
             "features is scrambled. Bigger means the model leans on it more.")
    names = sorted(imp["model"].unique())
    pick = st.selectbox("Model", names, index=names.index("Gradient-Boosted Trees")
                        if "Gradient-Boosted Trees" in names else 0)
    top = imp[imp["model"] == pick].sort_values("importance", ascending=False).head(15)
    st.altair_chart(
        alt.Chart(top).mark_bar(cornerRadiusEnd=4, height=14, color=MODEL_COLORS[0])
        .encode(x=alt.X("importance:Q", title="Increase in Log-Loss when scrambled"),
                y=alt.Y("name:N", sort="-x", title=None),
                tooltip=["name", alt.Tooltip("importance:Q", format=".4f"),
                         alt.Tooltip("delta_auc:Q", title="AUC drop", format=".4f")])
        .properties(height=26 * len(top)), width="stretch")


# --------------------------------------------------------------- view 5
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
    gender = c1.radio("Gender", ["male", "female"], horizontal=True)
    team_type = c2.radio("Level", ["international", "club"], horizontal=True)
    model_path = c2.text_input("Model path", "hdfs:///data/models/win_probability_gbt")

    if st.button("Estimate"):
        with st.spinner("Scoring in the Spark container..."):
            prob, how = None, None
            # Preferred path: the cluster. Spark will not start on Java 18+
            # ("getSubject is not supported") and Windows also needs
            # winutils.exe, so a local session is the fallback, not the default.
            try:
                prob = da.score_state_in_container(
                    score, wickets, balls_remaining, target, match_type,
                    overs_cap, model_path, gender=gender, team_type=team_type)
                how = "scored in the spark-master container"
            except Exception as container_exc:  # noqa: BLE001
                try:
                    import sys
                    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "ml"))
                    from pyspark.ml import PipelineModel
                    from pyspark.sql import SparkSession
                    import predict_live_state as pls

                    spark = SparkSession.builder.appName("whatif").getOrCreate()
                    model = PipelineModel.load(model_path)
                    row = pls.state_row(score, wickets, balls_remaining, target,
                                        match_type, overs_cap, gender=gender,
                                        team_type=team_type)
                    prob = pls.predict(spark, model, row)
                    how = "scored in a local Spark session"
                except Exception as local_exc:  # noqa: BLE001
                    st.error(f"Scoring failed. Container: {container_exc}")
                    st.error(f"Local Spark: {local_exc}")

            if prob is not None:
                st.metric("Chasing team win probability", f"{prob:.1%}")
                st.caption(
                    f"Needs {max(target - score, 0)} runs from {balls_remaining} balls "
                    f"with {10 - wickets} wickets in hand. ({how})")
