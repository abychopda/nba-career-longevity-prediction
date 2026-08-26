# Results log

Local figures are five-fold `GroupKFold` on `pid`, scored at **player level** in every case.
Random-ranking floor: **0.0337**.

| Exp | Approach | Local CV (player) | Leaderboard | Gap |
|-----|----------|-------------------|-------------|-----|
| 1 | LogReg, row-level fit, mean-pooled to player | 0.4743 ± 0.0292 | 0.54765 | +0.073 |
| 2 | LogReg, row-level fit, mean-pooled, + per-40 rates and age | 0.4799 ± 0.0254 | **0.55569** | +0.076 |
| 2c | LogReg, player-level aggregated — calibration only | 0.4521 ± 0.0267 | 0.52102 | +0.069 |
| 3 | LogReg, **no class weighting**, row-level, mean-pooled | **0.5092 ± 0.0295** | **0.59167** | +0.082 |
| 3x | XGBoost, no weighting — transfer check | 0.4980 ± 0.0327 | **0.59636** | +0.098 |

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


## Experiment 3 — model family × imbalance strategy

Nine configurations, identical folds, row-level fit with mean pooling throughout.

| | none | sqrt-balanced (5.4×) | balanced (28.7×) |
|---|---|---|---|
| **logreg** | **0.5092 ± 0.0295** | 0.4989 ± 0.0266 | 0.4796 ± 0.0257 |
| **histgbm** | 0.4777 ± 0.0291 | 0.4908 ± 0.0217 | 0.4759 ± 0.0414 |
| **xgboost** | 0.4980 ± 0.0327 | 0.4901 ± 0.0290 | 0.4806 ± 0.0310 |

**Hypothesis Rejected.** Boosting does not beat the linear baseline — the best boosted cell
(xgboost/none) trails logreg/none by 0.011, t = −1.22, 2/5 folds. Native missing-value handling
works but buys nothing the indicator columns were not already supplying.

**The finding is the class weighting, not the algorithm.** Dropping it:

| family | none − balanced | t | folds |
|---|---|---|---|
| logreg | **+0.0296** | **+6.59** | 5/5 |
| xgboost | +0.0174 | +3.74 | 5/5 |
| histgbm | +0.0018 | +0.22 | 2/5 |

Balanced weighting recentres a decision threshold; AUPRC never applies one. Experiments 1 and 2
inherited the reflex untested and lost about three AUPRC points to it. Verified not to be a
convergence artefact (60–73 iterations unweighted against 144–163 weighted, all converged) and it
holds on AUROC (+0.0045, t = 2.42) and top-25 precision (+0.032) as well as AUPRC.

Carried into Experiment 4: unweighted logistic regression, tuning from 0.5092 rather than 0.4796.


## Transfer check: does the local-to-leaderboard gap hold for a tree model?

Every leaderboard score so far comes from a logistic regression, so the +0.07 gap has only been
measured on a linear model. Tree ensembles cannot extrapolate beyond their training range, and
the competition set is four years later than any training season, so their gap has a specific
reason to differ. Submission 3x holds the weighting at none, so the only difference from
submission 3 is the model family.

Reading fixed before the score is seen:

| xgboost LB | implied gap | reading |
|---|---|---|
| ≈ 0.58 | +0.082 | trees transfer like the linear model; tune XGBoost on local CV in Experiment 4 |
| ≈ 0.55 | +0.05 | trees transfer worse across the four-year gap; local CV would mislead a tuned search |
| ≈ 0.61 | +0.11 | trees transfer better; the local comparison understates boosting |

**Observed: 0.59636, a gap of +0.098** — the third reading. The boosted model transfers better
than the linear one and finishes **ahead on the leaderboard despite losing locally**.

| model | local | leaderboard | gap |
|---|---|---|---|
| logreg/none | 0.5092 | 0.59167 | +0.082 |
| xgboost/none | 0.4980 | **0.59636** | **+0.098** |

Gaps to date: +0.073, +0.076, +0.069, +0.082 (all logistic regression) against +0.098 for the
boosted model. Both differences are small and rest on one draw each: the local gap was never
established (t = −1.22) and the leaderboard gap is +0.005 with no interval. The pair is best read
as equivalent, not as a reversal.

**A plausible mechanism, not an established one.** The prior behind this check was that trees
would transfer *worse*, since they cannot extrapolate beyond their training range. The opposite
appears to hold, and there is a reason it might: under covariate shift a linear model extrapolates
without limit into feature ranges it never saw, while a tree saturates at the boundary of its
training data. Plateauing may be the safer failure mode across a four-year gap.

**What this changes in Experiment 4.** Selecting purely on local cross-validation risks picking
the wrong family, because local CV appears to understate the boosted model by roughly 0.016 of
gap. The tuned best of *each* family is therefore submitted rather than the local winner alone.
That is two submissions instead of one, and it hedges exactly the risk this check identified.
