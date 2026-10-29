# CLAUDE.md - Ball-by-Ball Win Probability

Project context for anyone (human or Claude) picking this repo up. Records what is built, how to run it,
the traps already hit, and what is left. Numbers below were measured on this machine, not estimated.

## 1. What this project is

A distributed Big Data pipeline over the Cricsheet ball-by-ball dataset (22,818 matches, ~11.5M deliveries)
that produces a calibrated-ish, delivery-by-delivery **win probability for the chasing team**, plus
phase-wise bowler analytics and a Streamlit dashboard.

Stack: HDFS + YARN + Hadoop Streaming MapReduce + Hive + PySpark/Spark MLlib, run as a 10-container
Docker Compose cluster on a single Windows laptop. Hadoop 3.2.1, Hive 2.3.2, Spark 3.1.1 (BDE2020 images).

Why Docker: a real multi-service cluster on one machine, reproducible with one command. Native Hadoop/Spark
on Windows needs winutils shims and is painful.

## 2. Repo layout

| Path | Contents |
|---|---|
| `data/raw/all_json/` | 22,818 Cricsheet match JSONs, 3.8 GB (gitignored) |
| `hdfs_scripts/` | `00_init_hdfs_dirs.sh`, `01_classify_and_upload.py`, `01b_upload_from_manifest.sh` (new), `02_verify_upload.sh` |
| `etl/` | `flatten_matches.py`, `schema_utils.py` (JSON -> Parquet; also holds all HDFS path constants) |
| `mapreduce/` | `mapper.py`, `reducer.py`, `phase_utils.py`, `run_job.sh` (bowler economy by phase), `verify_with_spark.py` (independent check of the job) |
| `hive/` | `01..05_*.hql`, `run_all.sh` (database `cricket`) |
| `features/` | `match_state_features.py`, `player_form_features.py`, `build_training_table.py` |
| `ml/` | `train_win_probability.py`, `score_all_matches.py`, `predict_live_state.py` |
| `dashboard/` | `app.py`, `data_access.py`, `export_snapshots.sh`, `data_cache/` (gitignored TSV snapshots) |
| `docker/` | Custom Dockerfiles: nodemanager (python3), spark-master and spark-worker (numpy) |
| `docker-compose.yml`, `hadoop.env` | Cluster definition and Hadoop/YARN config |

## 3. Running it

```bash
docker compose up -d --build            # --build matters: 3 services are built, not pulled
export MSYS_NO_PATHCONV=1               # REQUIRED in Git Bash before any `docker exec` with a /path
docker exec namenode hdfs dfsadmin -safemode get   # wait for "OFF" before touching HDFS
```

Stages (full commands also in README.md). `SUBMIT` below is the Spark invocation that actually works:

```bash
SUBMIT="/spark/bin/spark-submit --driver-memory 6g \
  --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
  --py-files /workspace/etl/schema_utils.py"

docker exec namenode     bash    /workspace/hdfs_scripts/00_init_hdfs_dirs.sh
docker exec spark-master python3 /workspace/hdfs_scripts/01_classify_and_upload.py --dry-run   # writes manifest
docker exec namenode     bash    /workspace/hdfs_scripts/01b_upload_from_manifest.sh           # ~42 min (bind-mount cost)
docker exec spark-master $SUBMIT /workspace/etl/flatten_matches.py
bash mapreduce/run_job.sh
bash hive/run_all.sh
docker exec spark-master $SUBMIT /workspace/features/match_state_features.py   # then player_form_, build_training_table
docker exec spark-master $SUBMIT /workspace/ml/train_win_probability.py
docker exec spark-master $SUBMIT /workspace/ml/score_all_matches.py
docker exec hive-server beeline -u jdbc:hive2://localhost:10000/cricket -e "MSCK REPAIR TABLE win_probability_by_ball;"
bash dashboard/export_snapshots.sh
./venv/Scripts/python.exe -m streamlit run dashboard/app.py --server.port 8501
```

UIs: HDFS 9870 | YARN 8088 | Spark 8080 | HiveServer2 10000 (web 10002) | dashboard 8501.
Data lives in named Docker volumes, so `docker compose down` keeps HDFS data (do not add `-v`).

## 4. Verified state (measured)

| Item | Value |
|---|---|
| Matches in HDFS | 22,818 (19,703 limited-overs, 3,115 test) |
| `deliveries` rows / cols | 11,492,709 / 35 (5,986,260 limited-overs, 5,506,449 test) |
| `matches` / `player_registry` | 22,818 rows (26 cols) / 594,171 rows (3 cols) |
| MapReduce `bowler_economy_by_phase` | 33,107 rows x 11 columns (the last two are `dot_balls`, `dot_ball_pct`); all 12 format x phase combos populated; league-wide dot-ball rate 47.54% (2,744,127 of 5,771,895 legal balls) |
| Hive tables (db `cricket`) | matches, deliveries, player_registry, bowler_economy_by_phase, win_probability_by_ball, wpa_by_ball, win_probability_calibrated, model_comparison, feature_importance, matchup_lookup (the opponent, venue and context evaluations are Parquet only) |
| `match_state` / `player_form` / training table | 5,986,260 / 407,937 / 2,565,140 rows |
| Label balance | 50.8% / 49.2% (no imbalance problem) |
| Split | Chronological. Train = seasons before 2021/22 (1,363,580 rows); test = 2021/22 onward (1,201,560) |
| Model | `GBTClassifier` in a Spark ML pipeline; features are in-play state + rolling form + match_type + `is_female` + `is_international` (context features added in step 6b) |
| **Held-out metrics** | **AUC 0.9133, accuracy 0.8248, F1 0.8248, Brier 0.1201, Log-Loss 0.3727, ECE 0.0164** (before step 6b: AUC 0.9109, accuracy 0.8196, Brier 0.1222, Log-Loss 0.3780, ECE 0.0198; pre-6b outputs are kept in HDFS `/data/backup_pre6b`) |
| Scored output | 2,565,140 balls across 17,711 matches in `win_probability_by_ball` |
| WPA output | `wpa_by_ball`, 2,565,140 rows; 17,711 null WPA (first ball of each chase); WPA range -0.755 to +0.892, mean ~0 |
| Calibration | `win_probability_calibrated` (2,565,140 rows) + saved model `/data/models/win_probability_isotonic`. Split column: train 1,363,580 / calibration 564,849 / evaluation 636,711 |
| **Calibration effect** (evaluation seasons 2024-2026 only) | Raw -> Isotonic(+clip): Brier 0.1200 -> 0.1196, Log-Loss 0.3718 -> 0.3682, ECE 0.0188 -> 0.0108, AUC 0.9139 -> 0.9138. Platt: ECE 0.0187 (no help). Before 6b the calibrator gained more (Log-Loss 0.3756 -> 0.3708, ECE 0.0220 -> 0.0069): the model now does part of its job, but the calibrated ECE ended slightly HIGHER than before (0.0108 vs 0.0069) |

All three synopsis targets (AUC >= 0.82, Brier <= 0.18, Log-Loss <= 0.50) are met on held-out data.
Always quote the held-out figures above for "unseen data" claims (the scored table also contains training seasons).

**Calibration (figures in this paragraph are from BEFORE step 6b; see the step 6b bullet in the progress list for the current ones).** The raw GBT over-predicts the chasing team by ~4-6 points across the 45-85% range (predicted
0.55 vs actual 0.50) and did so in 9 of 10 held-out seasons: a systematic bias, not noise. It is now corrected by an
Isotonic calibrator (`ml/calibrate_win_probability.py`). Residuals on unseen seasons: mid-range gap about +/-0.5
points, but 70-90% still ~2 points optimistic and the 30-40% bucket ~2 points pessimistic. The raw column
`win_probability` and everything built on it (`wpa_by_ball`, the dashboard) are NOT yet switched to the calibrated one.
**Cause (found in step 6):** the over-optimism is concentrated in women's cricket (45-85% band: +16.7 points women's
international, +14.1 women's club, +3.8 men's international, -2.1 men's club) and women's cricket doubled its share of
rows (14% of training rows, 30% of held-out rows). The features do not say whether a match is men's or women's. One
global calibrator therefore fixes the average but cannot fix the segment gaps; adding `is_female` and `is_international`
to the model removes most of them (see step 6 findings).

**Model comparison** (`ml/compare_models.py`, table `model_comparison`). Identical held-out rows (1,201,560), same
preprocessing, hyperparameters fixed in advance and not tuned on the held-out seasons:

| Model | AUC | Brier | Log-Loss | ECE | Accuracy | Precision | Recall | F1 | Train |
|---|---|---|---|---|---|---|---|---|---|
| Gradient-Boosted Trees (maxIter 60, depth 5) | 0.9133 | 0.1201 | 0.3727 | 0.0164 | 0.8248 | 0.8162 | 0.8234 | 0.8198 | 73 s |
| Logistic Regression | **0.9140** | **0.1192** | **0.3698** | **0.0093** | **0.8262** | 0.8182 | **0.8241** | **0.8211** | 19 s |
| Random Forest (100 trees, depth 8) | 0.9054 | 0.1264 | 0.3974 | 0.0333 | 0.8186 | 0.8115 | 0.8143 | 0.8129 | 55 s |

(Figures above are the current ones, after step 6b added `is_female`/`is_international` to all three. Before 6b: GBT 0.9109 / 0.1222 / 0.3780 / ECE 0.0198, LR 0.9115 / 0.1219 / 0.3758 / 0.0203, RF 0.9044 / 0.1270 / 0.3981 / 0.0317.)

Precision/recall/F1 are for "chasing team wins" at threshold 0.5. Findings to state honestly:
- **LR is marginally the best and GBT is essentially tied** (Log-Loss gap 0.003; LR wins 8 of 10 held-out seasons on
  Log-Loss). Do NOT describe GBT as "the best model". Random Forest is clearly worst and pulls probabilities toward 0.5.
- The 45-85% over-optimism appears in **all three** models, so it is not a GBT quirk. CORRECTED in step 6: it was first
  put down to "an era shift", but it is mostly a missing-context effect (no gender/level feature) combined with
  women's cricket doubling its share of rows. See the step 6 findings below.
- Likely reason trees add little: the engineered features (required rate, run-rate difference, resources left) already
  carry the nonlinearity. Consistent with Wickramasinghe (2022) finding that feature engineering drives performance.
- GBT stays the model of record only because scoring, calibration, WPA and the dashboard are already built on it.
- Caveat: nothing was tuned. A GBT tuned on a validation slice of late TRAINING seasons (never the test seasons)
  might overtake LR. If tuning is done, keep that protocol.

**Feature importance** (`ml/feature_importance.py`, table `feature_importance`). Built-in importances were cross-checked
with group-wise permutation importance on the held-out seasons (scramble a whole group of related features jointly;
stable to within 0.0013 across 2 repeats; unpermuted baselines reproduce the model comparison). Rise in Log-Loss when
the group is scrambled (baselines after 6b: GBT 0.3727, LR 0.3698):

| Group | GBT | LR |
|---|---|---|
| Chase pressure (runs required, required rate, rate difference, runs per wicket needed) | +0.749 | +1.091 |
| Overs & wickets in hand (D/L-style: balls remaining, wickets down/remaining, resources left) | +0.021 | +0.107 |
| Player form (rolling striker/non-striker strike rate, striker average, bowler economy) | +0.019 | +0.017 |
| Who is playing (gender, level; added in 6b) | +0.017 | +0.024 |
| Scoreboard (current score, current run rate) | +0.007 | +0.029 |
| Match format | +0.002 | +0.003 |
| Recent momentum (runs/wickets in last 12 and 30 balls) | +0.002 | +0.002 |

(Rankings above are after 6b. Player form fell from +0.030 to +0.019 for GBT: part of what "form" was telling the model was who is playing. The earlier figures in the next lines predate 6b.)

Findings:
- Chase pressure dominates every method and model; scrambling it drops AUC from 0.911 to about 0.55.
- Player form is a real but modest contributor (second for GBT; about +0.015 AUC).
- Recent momentum and match format add almost nothing once the state is known. The synopsis lists momentum as a headline
  feature; do not oversell it.
- Native shares agree on the top group but not in detail (RF gives player form under 1%, GBT about 14%); permutation
  supports the GBT. Logistic Regression signs are sensible (more runs per wicket needed or a higher required rate lowers
  the win probability; more wickets in hand, more balls left or a higher rolling strike rate raise it).
- Caveats: group permutation measures a group's UNIQUE contribution while the others stay intact, so a group whose
  information is duplicated elsewhere is understated. The D/L-style group is mostly redundant with chase pressure, so its
  small GBT number must NOT be read as "overs and wickets do not matter". Whether the richer state beats an
  overs-and-wickets-only state needs a retrain ablation (not done). This is global importance only: there is no
  per-prediction explanation.

**Opponent-performance features** (`features/opponent_features.py`, `ml/evaluate_opponent_features.py`; tables under
`/data/processed/features/opponent/{batter,bowler,team}`). Six features: striker and bowler record against this opposing
team (ratio to their own overall rate in the same format family, shrunk toward 1.0 when the record is thin, plus the
number of earlier balls) and the batting team's earlier head-to-head win rate and meeting count. Built from earlier
matches only; verified against an independent brute-force recomputation (0 mismatches in 400 sampled rows per table,
about 250-330 of them with real history). Retrained with and without them, same split, same untuned hyperparameters:

| Model | Variant | AUC | Brier | Log-Loss | Change in Log-Loss (95% match-level CI) |
|---|---|---|---|---|---|
| GBT | baseline | 0.9108 | 0.1223 | 0.3781 | |
| GBT | + opponent features | 0.9128 | 0.1210 | 0.3752 | -0.0029 [-0.0052, -0.0006], helped 9 of 10 seasons |
| Logistic Regression | baseline | 0.9115 | 0.1219 | 0.3757 | |
| Logistic Regression | + opponent features | **0.9145** | **0.1198** | **0.3702** | -0.0055 [-0.0075, -0.0035], helped 10 of 10 seasons |

- The gain is real but small (about 0.8-1.5% in Log-Loss, +0.002 to +0.003 AUC) and is **concentrated early in the chase**
  (first quarter: -0.008 GBT, -0.010 LR; last quarter about 0). That is what team/opponent information should do: it is a
  prior that matters before the match state has revealed much, and it fades as the state takes over. A label leak would
  not fade like that.
- Calibration is not improved (ECE 0.0208 -> 0.0210 GBT, 0.0203 -> 0.0207 LR): the over-optimism remains (its real
  cause, missing gender/level context, was found in step 6).
- Logistic Regression again beats GBT, and gains more from the new features.
- **Not tested:** which of the three features does the work (the team head-to-head rate is the likeliest, since the gain is
  early). A per-feature ablation would settle it.
- The production model does NOT use the opponent or venue features (it does use the step-6 context pair since 6b). Adopting
  more means another consolidated re-run (scoring, calibration, WPA, comparison, importance).
- Row-order noise: Spark's GBT moves by about 1e-4 in Log-Loss when the same rows arrive in a different order (e.g. after a
  join). The baseline reproduces step 3 exactly on the plain table (difference 0.0). Compare variants on the same frame.

**Venue par features and the gender/level control** (`features/venue_features.py`, `ml/evaluate_venue_features.py`,
`ml/evaluate_context_features.py`; tables `/data/processed/features/venue`, `venue_feature_evaluation`,
`venue_feature_comparisons`). Grounds are keyed on venue AND city ("County Ground" alone is six grounds). Features:
shrunk first-innings par run rate, venue chase win rate, depth of history, and target-vs-par. Built from earlier matches
only; 0 mismatches against a brute-force recomputation. Coverage: 91% of matches have at least one earlier match at the
ground, 68% have five or more.

Change in Log-Loss vs baseline (negative = better; 95% match-level intervals):

| Comparison | Logistic Regression | GBT |
|---|---|---|
| + venue features | -0.0092 [-0.0113, -0.0071], 10/10 seasons | -0.0021 [-0.0045, +0.0003], not distinguishable |
| + venue features, shuffled across matches (placebo) | -0.0005 (tiny, from the target-run-rate arithmetic) | **+0.0029, WORSE** |
| + `is_female`, `is_international` only | **-0.0060 [-0.0086, -0.0034], 10/10** | **-0.0056 [-0.0077, -0.0036], 9/10** |
| venue features ON TOP of gender/level | -0.0026 [-0.0037, -0.0015], 9/10 | not run |
| + opponent + venue vs + opponent | -0.0088 [-0.0109, -0.0067], 10/10 | -0.0024, not distinguishable |

Findings:
- **Two plain columns, `is_female` and `is_international`, are the single most valuable addition so far.** For Logistic
  Regression they halve calibration error (ECE 0.0203 -> 0.0093) and cut the raw 45-85% gap for women's international
  from +18.3 to +1.5 points. The GBT also gains (-0.0056) but its segment gaps only shrink (+17.0 -> +6.1).
- About two thirds of the LR "venue" gain is just this gender/level context (the format norm behind venue par is split
  by gender and team_type). The genuinely venue-specific part is about -0.0026 for LR.
- The placebo is not a harmless control for the GBT: random match-level features HURT it (+0.0029), because trees can
  memorise individual matches. For the GBT, judge a feature set against the baseline, not against the placebo.
- Logistic Regression with opponent + venue reaches Log-Loss 0.3612 (AUC 0.9185, Brier 0.1162) vs GBT 0.3721 on the
  same features, so the gap to the GBT widened. Untuned GBT hyperparameters (60 rounds, depth 5) may be the reason.
- Not tested: whether the step-5 opponent gain survives once gender/level is in the baseline (`team_h2h_matches` could
  partly proxy competition level); GBT with venue ON TOP of gender/level.
- Consequence for step 2: the isotonic calibrator is a band-aid for a missing feature. If gender/level goes into the
  model, the calibrator must be re-fitted afterwards, and may be much less necessary.

**Batter vs bowler matchup lookup** (`features/matchup_lookup.py`, `hive/10_create_matchup_lookup_table.hql`, table
`matchup_lookup`, 1,168,224 rows). A browsable lookup, NOT a model feature. Per (format family, batter, bowler) plus pooled
`ALL` rows: balls, runs, dismissals, dots, fours, sixes, strike rate, dot %, tier, an expected strike rate from the two
players' overall quality (batter rate x bowler rate / league rate, each excluding the pairing), a shrunk strike rate and
a z-score. Reconciles exactly with the source (5,796,419 balls, 5,853,141 runs, 232,772 dismissals); a sampled recompute
has 0 mismatches; 11 balls dropped where a same-name id collision made a player face himself.

How thin the data is (the reason it is a lookup):
- T20: 83% of 421,839 pairings are "thin" (under 12 balls) and hold half of all balls; only 2.8% of pairings have 30+ balls.
- ODI: 61% of 191,444 pairings are thin; 10% have 30+ balls.
- Even with 30+ balls, 21% of T20 pairings and 43% of ODI pairings have never produced a dismissal, so dismissal-based
  figures are unreliable for most pairings.

Is there matchup-specific signal? Among pairings with 30+ balls, z (observed vs two-player expectation) has variance
1.72 (T20) and 1.61 (ODI), and 12.6% / 11.4% fall beyond +/-2 against 4.6% expected by chance. So there is more spread
than noise produces, but this is an UPPER bound on a true matchup effect: it also absorbs phase of the innings, venue and
match-to-match form (balls within one meeting are correlated). For a pairing flagged |z| > 2, roughly 1 in 3 (4.6/12.6)
would be flagged by chance alone. Treat flags as prompts to look, not conclusions. Settling it needs an out-of-sample
test (does the earlier record predict later meetings beyond the expectation?), which has NOT been run.
- The table summarises the whole dataset "as of now"; using it to predict a match inside that period would leak.

## 5. Progress - done

- Docker/WSL2 working (needed Virtual Machine Platform enabled; firmware was fine). `~/.wslconfig`: 10 GB / 6 CPU / 4 GB swap.
- All 7 pipeline stages run end to end, every hop verified by row counts.
- Cluster is reproducible: `docker compose down` then `up -d --build` re-verified (stage 3 and model scoring rerun clean).
- Dashboard with 5 views (step 9): bowler economy by phase (economy or dot-ball %), match explorer (win-probability curve, WPA per ball,
  five biggest swings), batter vs bowler (matchup lookup), model scorecard (comparison, reliability chart, permutation importance), what-if scorer.
  The what-if view scores via `docker exec spark-master` (host has Java 26, which Spark cannot run on); local Spark is only a fallback.
- README corrected to match what actually works.
- **Win Probability Added** (`ml/compute_wpa.py`, `hive/06_create_wpa_table.hql`): per-delivery P_t - P_{t-1} from the chasing
  team's view. Verified: row count equals the scored table, one null per chase, zero duplicate ball keys, per-chase WPA
  sums equal last-minus-first probability exactly. Run it after stage 6, and create the Hive table after the job.
- **Calibration correction** (`ml/calibrate_win_probability.py`, `hive/07_create_win_probability_calibrated_table.hql`).
  Fit on the first 5 held-out seasons, judged on the last 5 (never used for fitting). Cross-checked against an
  independent scikit-learn fit (identical to 4 decimals); the saved model reloads and reproduces every stored value exactly.
  Design notes: (a) fitting on training seasons would learn nothing because the GBT is already calibrated there;
  (b) isotonic is a step function whose ends hit exactly 0.0 and 1.0 (9 of 2,639 "0%" rows actually won), so outputs
  are clipped to [0.005, 0.995], which is metric-neutral (Log-Loss 0.3707 vs 0.3708); (c) the `split` column marks
  which rows are valid for judging: only `evaluation`. Do not quote calibration metrics from `train` or `calibration` rows.
- **Model comparison** (`ml/compare_models.py`, `hive/08_create_model_comparison_table.hql`): GBT vs Logistic
  Regression vs Random Forest on identical held-out rows. Retrained GBT reproduces the saved run (AUC difference
  4e-7) and its Brier/Log-Loss match the independent pandas figures. LR and RF models saved to HDFS.
- **Feature importance** (`ml/feature_importance.py`, `hive/09_create_feature_importance_table.hql`): native importances for
  GBT, Random Forest and Logistic Regression (standardised, signed coefficients) plus group-wise permutation importance for
  GBT and LR on the held-out seasons. 96 rows written and verified through Hive (107 after 6b added a group).
- **Opponent-performance features** built and evaluated (see section 4). Leakage-checked against a brute-force
  recomputation; the check also caught one duplicated (match, batter) key caused by two same-name players on opposite
  sides, which is now excluded. Result: a small, real improvement concentrated early in the chase. Not adopted into the
  production model yet.
- **Venue par features** built, leakage-checked, and evaluated with a placebo and a gender/level control (see section 4).
  The control found that the model lacks gender/level context, which explains most of the calibration problem and most
  of the "venue" gain. The earlier "era shift" explanation of the over-optimism was wrong and has been corrected.
- **Batter vs bowler matchup lookup** built and reconciled (see section 4): 1,168,224 rows with sample-size tiers, a
  two-player expected strike rate and z-scores, queryable through Hive. Shown in the dashboard's batter-vs-bowler view.
- **Dot-ball percentage in the MapReduce job** (`mapper.py`, `reducer.py`, `hive/04`). Two columns appended to the end of the
  output (`dot_balls`, `dot_ball_pct`), so no earlier column moved. Proof: the original nine columns are byte-for-byte
  identical to the pre-change output (matching checksums, backup was kept in `/data/mapreduce_backup/economy_before_dots`),
  and `mapreduce/verify_with_spark.py` recomputes all 33,107 rows in Spark from the deliveries Parquet with zero mismatches on
  runs, balls, wickets, dots and every derived column. League-wide dot-ball rates (dot %): ODI powerplay 68.1, middle 54.0,
  death 40.7; T20 powerplay 50.8, middle 37.2, death 34.0. Two bowler-phase rows have no legal balls, so their percentage is
  blank. Also fixed an inconsistency from the matchup step: `matchup_lookup.dot_pct` was a fraction and is now a percentage.
- **Context features adopted (step 6b)** (`ml/train_win_probability.py`: `CONTEXT_FEATURES`, derived in `prepare()` from the
  training table's `gender` and `team_type`, which have no nulls; `ml/verify_context_adoption.py`). Backup of every output
  taken first in HDFS `/data/backup_pre6b` (identical file counts and bytes). Re-run in order: train, score, compare_models,
  feature_importance, calibrate, compute_wpa, Hive tables, dashboard snapshots. Results on the same 1,201,560 held-out rows:
  Log-Loss 0.3780 -> 0.3727 (paired change -0.00533, match-level 95% interval [-0.00729, -0.00336], 9 of 10 seasons better,
  2024 slightly worse), Brier 0.1222 -> 0.1201, ECE 0.0198 -> 0.0164, AUC 0.9109 -> 0.9133. Raw 45-85% calibration gap by
  segment (predicted - actual), before -> after: women's club +0.141 -> +0.052, women's international +0.167 -> +0.067,
  men's club -0.021 -> -0.003, men's international +0.038 -> +0.037 (unchanged). So the gap is cut by about 60% for women
  but not gone, and men's international is untouched. Calibrator: still helps (Log-Loss 0.3718 -> 0.3682) but less, and its
  evaluation-season ECE (0.0108) is slightly worse than before 6b (0.0069). The scored and calibrated tables cover exactly
  the same 2,565,140 rows as before with the same split labels. Dashboard what-if view now has gender and level selectors
  (same chase state: 0.346 men's international, 0.234 women's international). The step 5 and 6 evaluation scripts still use
  their own pre-6b recipe, so their "baseline" means the pre-6b model.
- **Dashboard extension (step 9)** (`dashboard/app.py`, `data_access.py`, `export_snapshots.sh`). New snapshots: `model_comparison`,
  `feature_importance`, `calibration_buckets` (reliability table aggregated in Hive on the `evaluation` split), `wpa_by_ball`
  (slim: match_id, ball_seq, prev prob, wpa; 2,547,429 rows) and `matchup_lookup` (balls >= 12 only: 290,684 of 1,168,224 rows).
  Every snapshot row count was checked against Hive. All views run without exceptions under Streamlit AppTest and were looked
  at in a browser. Only the CSV path was exercised: PyHive is not installed on this host, so the Hive-side SQL added to
  `data_access.py` is written but untested. Findings: (a) the economy ranking is dominated by strong-side bowlers against weak
  associate sides (e.g. T20 powerplay economy 3.6 with 68% dots; checked against raw deliveries, real data, not a pipeline
  bug) so the view carries a warning; (b) Altair `sort="x"` on a layered chart did not order the bars, so the order is now explicit.
  The calibrated probability is shown only as an aggregate reliability chart, not per match (would need another ~100 MB snapshot).
- **Full project report** (`docs/PROJECT_REPORT.md`, Word copy `docs/Project_Report.docx`): evidence-based 14-chapter report, status table, testing report, demo script, slide outline, 25 viva Q&A, readiness assessment and the list of information still needed from the team. Numbers in it are the post-6b ones.
- Documents produced (outside the repo): synopsis, literature survey table, project charter (docx + published artifact).

### Bugs found and fixed (read before debugging similar symptoms)

1. `hadoop.env` had no `yarn.nodemanager.aux-services` -> no MapReduce job could run.
2. NodeManager memory 4096 MB < AM + reduce container -> job would hang forever at reduce. Now 8192.
3. `run_job.sh`: `-files` placed after `-partitioner`; generic options must precede streaming options.
4. `run_job.sh`: needs `-cmdenv PYTHONPATH=.` - symlinked task files break `import phase_utils` (fails only under YARN).
5. `hive/run_all.sh` had a literal `\n` -> loop hit a bogus filename and stage 05 never ran.
6. `MSCK REPAIR` ordering: Hive runs before ML writes, so `win_probability_by_ball` showed 0 rows silently. Repair after stage 6.
7. beeline `csv2` wraps comma-containing values in NUL bytes (corrupted 56% of `matches` rows). Exports are TSV now.
8. Momentum features were never imputed, so the what-if scorer failed on nulls. Now in the `Imputer` (metrics unchanged).

## 6. What is left (prioritised)

Status key: cheap = hours, medium = a day or two, risky = needs new software or has data-quality limits.

| # | Item | Effort | Notes |
|---|---|---|---|
| 1 | ~~Win Probability Added (WPA)~~ **DONE** | done | `ml/compute_wpa.py` + `wpa_by_ball` table. Shown in the dashboard's match explorer. |
| 2 | ~~Calibration correction~~ **DONE** | done | Isotonic, fit/judged on a time split of the held-out seasons. Open follow-up: decide whether WPA and the dashboard should use the calibrated column (isotonic gives only ~715 distinct values, so calibrated WPA has flat stretches of exactly 0). |
| 3 | ~~Model comparison (LR + RF)~~ **DONE** | done | See section 4. Open follow-ups: SVM / Decision Tree / Voting (Chakraborty et al. list) are not built; optional validated tuning of GBT. |
| 4 | ~~Feature importance~~ **DONE** | done | Native + group permutation; see section 4. Optional follow-up: a retrain ablation (overs, wickets, runs required only vs the full feature set) to test whether the richer state beats a D/L-style state. |
| 5 | ~~Opponent-performance features~~ **DONE (evaluated, not adopted)** | done | Small real gain; see section 4. Open: which of the three features does the work, and whether to adopt them in a single consolidated re-run at the end. |
| 6 | ~~Venue par baselines~~ **DONE (evaluated, not adopted)** | done | Small venue-specific gain beyond gender/level; see section 4. The bigger discovery is the missing gender/level context. |
| 6b | ~~Add `is_female`, `is_international` to the model~~ **DONE** | done | Done and verified (progress list, step 6b): Log-Loss -0.0053 with a match-level interval excluding zero; women's calibration gap cut ~60% but not removed. Open: men's international gap unchanged; opponent/venue features still not adopted. |
| 7 | ~~Batter-bowler matchup lookup~~ **DONE** | done | `matchup_lookup` table with reliability tiers; see section 4. Open: an out-of-sample test of whether earlier head-to-head predicts later meetings Dashboard view done. Still NOT a model feature. |
| 8 | ~~Dot-ball % in the MapReduce job~~ **DONE** | done | `dot_balls`, `dot_ball_pct` appended to the output and the Hive table; verified (section 4). Shown in the dashboard's economy view. |
| 9 | ~~Dashboard extension~~ **DONE** | done | Five views, see section 4. Open: Hive-path SQL untested; no player-name lookup for the economy view (bowler_id only); no gender/level filter. |
| 10 | XGBoost / LightGBM | risky | Not in MLlib. Needs xgboost4j-spark or SynapseML on Spark 3.1.1 (JAR compatibility risk). |
| 11 | LSTM | risky | Not in MLlib; needs TensorFlow/PyTorch, per-match sequence data, CPU-only. Treat as optional. |
| 12 | TreeSHAP | risky | Depends on #10. Global feature importance (#4) is delivered; only the per-prediction explanation is still absent. |

Agreed fallback if time runs short: the three-model comparison stands as delivered; XGBoost/LightGBM and LSTM are
stretch goals, not requirements. (GBT is the model of record for pipeline continuity, not because it won.)

### Housekeeping and open decisions

- **Everything since commit `1ba3683` is uncommitted.** Delete `*.bak*` and `verify_*.txt` first, then commit
  the real changes. These files are not in `.gitignore`.
- **No git remote exists** - the code lives only on this laptop. Push to GitHub/GitLab if others must see it.
- Delta Lake (listed in an earlier plan) adds little here: the data is immutable after load. Recommend dropping it.
- The synopsis does not mention the dashboard, but it is built and working; decide whether to describe it.
- Team matrix names are still "to be assigned".
- Decide whether to adopt gender/level (and the opponent and venue features) into the production model in ONE
  consolidated re-run, then re-fit the calibrator. Steps 5 and 6 evaluated features without changing production.
- Wherever the over-optimism is explained (report, viva), use the corrected cause (missing gender/level context plus
  women's cricket doubling its share), not "era shift".
- The published artifact and the first Word charter quote Brier 0.120 / Log-Loss 0.373 and describe them as measured on
  unseen seasons. Held-out values are 0.122 / 0.378 (section 4); update those if they are going to be presented.

## 7. Design invariants - do not break these

- **No leakage.** Train/test split is by date, never random. Rolling player form uses
  `rowsBetween(-10, -1)` so the current match never contributes to its own average.
- **Verify by row count, not exit code.** Several failures here exited 0 (0 rows, empty tables, header-only exports).
- **`match_id` is a string** (the filename stem; `wi_201706` must not collide with `201706`). Never cast it to int.
- Joins key on `(match_id, player_name)`: Cricsheet has no global player file, only per-match registries.
- Phase/model work covers **limited-overs only** (ODI/ODM/T20/IT20). Tests are stored and queryable but excluded.
- Phase boundaries (0-indexed overs): T20/IT20 0-5 / 6-14 / 15+; ODI/ODM 0-9 / 10-39 / 40+. Rain-shortened matches keep
  full-length boundaries (documented approximation).
- Bowler economy excludes byes/legbyes, charges wides/no-balls, and wides/no-balls are not legal balls.
- A bowler's dot ball is a LEGAL ball on which he is charged no runs (so a bye ball IS a dot; some sources disagree).
  `matchup_lookup.dot_pct` is the batter's view (balls faced with no runs at all) and differs by design. Both are
  percentages on a 0-100 scale; never mix a fraction and a percentage in one project.
- Team names are not unique: "India" men and "India" women share a name. Any team-level history must be split by
  `gender` and `team_type`. Player ids can also collide when two same-name players are on opposite sides of one match.
- Any history feature must be verified against a brute-force recomputation (a self-join on strictly earlier matches,
  no windows), not just assumed correct because the window ends at row -1.

## 8. Environment quirks (Windows host)

- Git Bash rewrites `/workspace/...` into `C:/Program Files/Git/workspace/...`. Always `export MSYS_NO_PATHCONV=1`.
- Container images are old: `spark-submit` is at `/spark/bin/spark-submit` (not on PATH); the Spark image has no
  `core-site.xml` (pass `fs.defaultFS`); namenode has `hdfs` but only Python 3.5; spark-master has Python 3.7 but no `hdfs`.
  That is why stage 1 is split across two containers.
- Host Java is 26, which Spark cannot run on. PyHive's `sasl` needs MSVC to build, so the dashboard uses TSV snapshots
  and says so in its sidebar.
- Bind-mounting the project into containers makes small-file work slow (stage 1 upload ~42 min).
- Host has no Node, pandoc or LibreOffice. Word files are built with `python-docx` in `venv`. Keep generator scripts
  pure ASCII (use `\uXXXX` escapes) and never print non-cp1252 characters to the console.
- `docker compose` no longer needs the `version:` key (removed).
- Spark is **3.1.1**: functions added later (`min_by`/`max_by` are 3.3+) do not exist. Use struct min/max instead.
- `beeline -f` with `MSCK REPAIR` prints a long run of `null` lines. Harmless; filter it when scripting.
- `F.col(...)` cannot be built at import time (no Spark session yet). Wrap such expressions in functions.
- When patching files from a shell heredoc, backslashes in the patch text get altered (a `\t` arrives as a real tab).
  Build them with `chr(92)` and verify the patched file by printing it. Some repo files have CRLF endings
  (`mapper.py`, `phase_utils.py`, most of `ml/`): patch them with a script that preserves CRLF, not a whole-file rewrite.
- Docker Desktop has been found stopped at the start of several sessions, so check `docker info` first and launch
  `%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe` if needed. Containers restart on their own afterwards.

## 9. Context for the evaluation

Graded on: Problem Definition 10%, Objectives and Outcomes 10%, Methodology 10%, Literature Survey 20%,
Progress 10%, Individual Understanding 20%, Continuity 20%. Honest framing matters more than breadth: the
synopsis promises multi-model comparison, calibration, and explanation, which items 1-4 above deliver cheaply.
Chakwate & Madhan (LSTM) and Chakraborty et al. (6-algorithm comparison) are cited gaps. Three models are now
compared (GBT, Logistic Regression, Random Forest); SVM, Decision Tree, XGBoost and a Voting ensemble are not built and
there is no LSTM, so do not claim parity with those papers' full technique lists until items 10 and 11 exist.
