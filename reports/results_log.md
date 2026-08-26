# Results log

Local figures are five-fold `GroupKFold` on `pid`, scored at **player level** in every case.
Random-ranking floor: **0.0337**.

| Exp | Approach | Local CV (player) | Leaderboard | Gap |
|-----|----------|-------------------|-------------|-----|
| 1 | LogReg, row-level fit, mean-pooled to player | 0.4743 ± 0.0292 | 0.54765 | +0.073 |
| 2 | LogReg, row-level fit, mean-pooled, + per-40 rates and age | 0.4799 ± 0.0254 | **0.55569** | +0.076 |
| 2c | LogReg, player-level aggregated — calibration only | 0.4521 ± 0.0267 | 0.52102 | +0.069 |

Row 2c is not a candidate model. It is the rejected arm, submitted to measure whether the
local-to-leaderboard gap depends on the training unit.

## Experiment 2 — training unit × model family

Four cells, identical folds, every cell scored at player level.

| | logistic regression | gradient boosting |
|---|---|---|
| **row-level fit + mean pool** | **0.4799 ± 0.0254** | 0.4750 ± 0.0396 |
| **player-level aggregated** | 0.4521 ± 0.0267 | 0.4561 ± 0.0452 |

Paired per-fold contrasts, four degrees of freedom:

| contrast | mean | t | folds |
|---|---|---|---|
| aggregating, under logreg | −0.0278 | −1.86 | 1/5 |
| aggregating, under histgbm | −0.0189 | −2.04 | 1/5 |
| histgbm − logreg, row-level | −0.0050 | −0.53 | 2/5 |
| histgbm − logreg, player-level | +0.0040 | +0.23 | 2/5 |
| **interaction (the hypothesis)** | **−0.0090** | **−0.53** | 3/5 |

**Hypothesis Rejected.** The unit effect does not depend on the model family — the interaction
contrast changes sign across folds. Aggregation loses under both families because it halves the
training set (37,703 seasons → 17,938 players; 1,271 positive rows → 604 positive players) to buy
career information that pooling already captures.

Nothing here is significant at 4 df. What is consistent is direction: aggregation won 2 of 10
fold-comparisons, and the leaderboard agreed (0.55569 row-level against 0.52102 player-level).

## Pooling rule

Experiment 1 tested mean, max and median. Final-season pooling was never tested and is the most
intuitive rule of the lot, so it was added here.

| rule | AUPRC | vs mean | t | folds |
|---|---|---|---|---|
| **last** | **0.4812 ± 0.0264** | +0.0013 | +0.16 | 3/5 |
| mean | 0.4799 ± 0.0254 | — | — | — |
| median | 0.4774 ± 0.0267 | −0.0025 | −0.38 | 2/5 |
| max | 0.4555 ± 0.0312 | −0.0244 | −2.47 | 0/5 |
| min | 0.4554 ± 0.0191 | −0.0246 | −6.67 | 0/5 |
| first | 0.4181 ± 0.0228 | −0.0618 | −4.90 | 0/5 |

**Recency is real; the pooling rule is not the way to exploit it.** Scoring a player on their
final season alone beats scoring them on their first by 0.063, a large and consistent effect.
But mean pooling already captures it — last and mean are indistinguishable at +0.0013, t = +0.16.
Mean is retained on the grounds that it uses every prediction rather than one, which is the
safer default for the 63.3% of players with more than one season. Switching to last-season
pooling would be defensible and would change nothing measurable.

## Local-to-leaderboard calibration

Three observations, two framings:

| submission | local | leaderboard | gap |
|---|---|---|---|
| Exp 1, row-level | 0.4743 | 0.54765 | +0.073 |
| Exp 2, row-level | 0.4799 | 0.55569 | +0.076 |
| Exp 2, player-level | 0.4521 | 0.52102 | +0.069 |

The gap is stable at +0.07 across both framings, so it belongs to the pipeline rather than the
training unit. Two consequences: local cross-validation is a conservative but faithful predictor
of leaderboard position, and the local comparison between the two framings was a fair one — the
leaderboard reproduced its ordering and roughly its magnitude.

Experiment 2 beat Experiment 1 by +0.008 on the leaderboard against +0.006 locally. Consistent,
but a single draw on each and well inside the ±0.025 fold spread, so it is not established.
