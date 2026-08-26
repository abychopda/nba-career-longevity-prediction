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
| 4 | LogReg, untuned, redundancy-pruned feature set | 0.5068 ± 0.0298 | 0.58688 | +0.080 |
| 4x | XGBoost, untuned, redundancy-pruned feature set | 0.5006 ± 0.0317 | 0.58210 | +0.081 |
| 4b | Rank average of both families, full set, mean pooling | 0.5147 ± 0.0329 | 0.60795 | +0.093 |
| 4c | Rank average of both families, full set, final-season pooling | **0.5182 ± 0.0372** | **0.62238** | +0.104 |

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


## Experiment 4 — feature selection, recruiting-rating ablation and tuning

Two selection screens measured against the full matrix on identical folds, both families untuned.

| set | columns | logistic regression | XGBoost |
|---|---|---|---|
| full | 108 | **0.5092 ± 0.0295** | 0.4980 ± 0.0327 |
| redundancy-pruned | 98 | 0.5068 ± 0.0298 | **0.5006 ± 0.0317** |
| gain-ranked top 20 | 73 | 0.4676 ± 0.0254 | 0.4880 ± 0.0250 |

Paired against the full set, four degrees of freedom:

| contrast | mean | t | folds |
|---|---|---|---|
| pruned, under logistic regression | −0.0024 | −0.96 | 2/5 |
| pruned, under XGBoost | +0.0027 | +0.58 | 3/5 |
| gain-ranked top 20, under logistic regression | **−0.0417** | **−13.94** | 0/5 |
| gain-ranked top 20, under XGBoost | −0.0099 | −1.17 | 1/5 |

**The pruned set is adopted on parsimony**, being statistically indistinguishable from the full
set under both families and ten columns cheaper. The gain-ranked set is rejected, and its
asymmetry is the informative part: a linear model spreads its prediction across many weak columns
and loses 0.042 when they go, while a boosted ensemble already concentrates on the top of its own
ranking and loses a quarter as much.

The screens differ in kind, not only in strength. Redundancy pruning never reads the target, so it
cannot overfit and needs no fold discipline; gain ranking is target-aware and was therefore
computed inside each fold on its own training rows only.

## The recruiting-rating ablation

On the pruned set, removing the two inputs that are not college performance.

| removed | logistic regression | XGBoost |
|---|---|---|
| nothing | 0.5068 | 0.5006 |
| recruiting rating and `is_ranked` | 0.5084 (+0.0015, t = +0.62, 3/5) | 0.4881 (−0.0125, t = −1.60, 2/5) |
| `age_at_season` | 0.5009 (−0.0060, t = −1.07, 2/5) | 0.4915 (−0.0091, t = −1.99, 1/5) |
| both | 0.5027 (−0.0041, t = −0.97, 2/5) | 0.4856 (**−0.0150**, t = **−4.97**, 0/5) |

**The ranking's dependence on high-school scouting data is small, and under the linear model not
measurable at all.** Removing the recruiting rating costs logistic regression nothing — +0.0015,
the wrong sign for a dependency — and XGBoost 0.0125, which does not clear the noise floor at four
degrees of freedom. Removing both non-college inputs costs the boosted model 0.0150 with every
fold agreeing, the only ablation contrast that is consistent.

This is the number the report's scope justification and its fairness section both need: a club
unwilling to inherit recruiter judgements can have a ranking built on college performance alone
for roughly 0.012 AUPRC, and under a linear model for nothing.

## Hyperparameter tuning — a measured null

Logistic regression's 18-configuration grid searched exhaustively; XGBoost drew 20 of 648. Both
scored on three folds inside a search partition of four fifths of the players, with the remaining
fifth never scored during the search.

| family | defaults, search folds | best of search | apparent gain | untouched holdout | full five folds |
|---|---|---|---|---|---|
| logistic regression | 0.4807 | 0.4836 | +0.0028 | 0.5457 → 0.5417 (−0.0039) | +0.0000, t = 0.00 |
| XGBoost | 0.4474 | 0.4735 | **+0.0260** | 0.5571 → 0.5569 (−0.0002) | +0.0012, t = 0.18 |

**XGBoost's search appeared to gain 0.026 and delivered −0.0002 on players it had never scored.**
That is what a validation artefact looks like once it is measured outside the folds that chose it,
and it is why the holdout was reserved before the search rather than after seeing the result.

Chosen configurations: logistic regression `C=0.1, penalty=l2`; XGBoost `n_estimators=150,
learning_rate=0.06, max_leaves=31, min_child_weight=20, subsample=0.7, colsample_bytree=0.5,
reg_lambda=5`. A confound worth naming: search folds fit on 53% of the rows against 80% for the
final folds, so the search may favour configurations that are over-regularised at full training
size. That weakens the search rather than the conclusion, and the holdout confirms it
independently.

## Two preprocessing reflexes, re-checked at the tuned settings

| check | logistic regression | XGBoost |
|---|---|---|
| unweighted advantage over balanced weighting | +0.0250, t = 7.88, 5/5 | +0.0039, t = 0.72, 3/5 |
| cost of rescaling columns to the unit interval | already scaled | −0.0013, t = −1.30, 2/5 |

Unweighted fitting stays ahead in both families, decisively for the linear model and inside noise
for the boosted one, so nothing revives the case for weighting the classes. Rescaling the boosted
arm is measured across five folds rather than on one split, because a single split moves by more
than the fold spread — XGBoost builds its histogram in float32, so an affine rescale that is not
an exact power of two perturbs its bin edges. Doubling every column, which is exact in binary
floating point, leaves the predictions bit-identical.

**Top of the ranking, on the holdout** (base rate 0.0351): AUPRC 0.5457, AUROC 0.9044 as a
secondary diagnostic; top 25 precision 0.880 (25×), top 50 0.760 (22×), top 100 0.600 (17×).

**Hypothesis Rejected.** The tuning gain does not survive the holdout, and the reduced feature set
does not survive the leaderboard — see the next section. Neither family's tuned configuration
clears the ±0.025 threshold this project treats as an established change, so both submissions
carried the untuned settings on the pruned set, which made the pair a direct leaderboard test of
the pruning.


## Leaderboard outcome: the reduced feature set does not transfer

Experiment 4's two submissions differed from Experiment 3's only by the ten pruned columns, which
made the pair a clean single-variable test.

| model | local full | LB full | local pruned | LB pruned | leaderboard cost |
|---|---|---|---|---|---|
| logistic regression | 0.5092 | 0.59167 | 0.50685 | 0.58688 | **−0.00479** |
| XGBoost | 0.4980 | **0.59636** | 0.50061 | 0.58210 | **−0.01426** |

Pruning cost nothing measurable on five folds and 0.005 / 0.014 on the competition set. The gaps
locate the damage:

| model and feature set | local | leaderboard | gap |
|---|---|---|---|
| XGBoost, full | 0.4980 | 0.59636 | **+0.0984** |
| XGBoost, pruned | 0.50061 | 0.58210 | **+0.0815** |
| logistic regression, full | 0.5092 | 0.59167 | +0.0825 |
| logistic regression, pruned | 0.50685 | 0.58688 | +0.0800 |

**XGBoost's gap fell by 0.0169; the linear model's moved 0.0024.** Pruned XGBoost now transfers at
about the linear model's rate. The two families' full-set leaderboard scores differ by 0.0047 on
one draw each, so the family ordering is not itself established; what is measured here is the
change in each family's own gap.

**A candidate mechanism, not a tested one.** Under covariate shift a tree can fall back on a
near-duplicate column when a feature's distribution drifts across the four years to the
competition seasons, and pruning removes that fallback. This is the most likely reading of two
submissions: the pruning threshold was never varied and no individual column was traced, so the
mechanism is not established.

**Why cross-validation could not catch it.** This does not depend on the mechanism. Every fold is drawn from 2008–2016, so redundancy costs
nothing there and any effect appearing only under the four-year shift cannot show up locally,
whatever its cause. It is the second time in this project that local cross-validation has been
uninformative on a question about transfer, after Experiment 3's finding on the boosted family.

**Consequence: the full 108-column set stands.** The ablation and tuning conclusions are
unaffected, since both were measured on the same folds under the same conditions as everything
they are compared against.

## Combining the two families

Neither family dominates: the linear model wins locally by 0.011 and the boosted one won the
leaderboard by 0.005, which is the signature of two models wrong about different players. A rank
average needs no fitted weight, is insensitive to the two models' different probability scales,
and unlike pruning removes nothing.

Measured on the same five folds, full feature set, untuned settings:

| candidate | local AUPRC | vs XGBoost | vs logistic regression |
|---|---|---|---|
| rank average, final-season pooling | **0.5182 ± 0.0372** | +0.0202, t = +2.77, 5/5 | +0.0089, t = +1.25 |
| logistic regression, final-season pooling | 0.5149 ± 0.0331 | +0.0169, t = +1.31, 5/5 | +0.0057, t = +0.80 |
| **rank average, mean pooling** | **0.5147 ± 0.0329** | +0.0168, t = +2.91, 4/5 | +0.0055, t = +1.16 |
| logistic regression, mean pooling | 0.5092 ± 0.0295 | +0.0113, t = +1.22, 3/5 | — |
| XGBoost, mean pooling | 0.4980 ± 0.0327 | — | — |
| XGBoost, 400 trees at learning rate 0.04 | 0.4966 ± 0.0287 | −0.0013, t = −0.32 | — |

The rank average beats both parents. Its margin over XGBoost is one of the more consistent in
the project at t = +2.91 on 4/5 folds, though well short of the class-weighting result in
Experiment 3 (t = +6.59, 5/5), and its margin over the linear model at +0.0055 is not
established. A deeper, slower boosted ensemble is a dead end.

**Reading fixed before the scores were seen.** At or above 0.605 the ensemble transfers and
becomes the final model; between 0.596 and 0.605 it matches the best single model with no
established gain; below 0.596 blending fails to transfer and Experiment 3's XGBoost stands.

One caveat on the files: AUPRC reads only the order of predictions, so a rank average is a
legitimate submission, but its values are normalised ranks rather than calibrated probabilities.
Anything downstream needing a probability should use one of the single models.


## Leaderboard outcome: the ensemble transfers

| submission | local | leaderboard | gap |
|---|---|---|---|
| Experiment 3, XGBoost, full set | 0.4980 | 0.59636 | +0.098 |
| rank average, mean pooling | 0.51472 | 0.60795 | +0.093 |
| **rank average, final-season pooling** | 0.51816 | **0.62238** | **+0.104** |

Both clear the 0.605 fixed in advance, so the ensemble becomes the final model at **+0.026** over
the best single model. The local reading understated it: the ensemble's margin over the linear
model was +0.0055 with a t of 1.16, which this project's own ±0.025 rule would not have called
established. Acting on it was justified by the mechanism — two models wrong about different
players — rather than by the size of the local margin, and that is the honest reason to give.

**Final-season pooling is worth +0.014 on the competition set against +0.0035 on the folds.** That
is the third time a difference the folds could not resolve has mattered on the leaderboard, after
the boosted family in Experiment 3 and the redundant columns above. Each is a single draw and none
is established on its own. What they share is a direction and a candidate reason: all three concern
behaviour under a four-year shift that same-period folds cannot exhibit.

The honest summary of this project's validation strategy is mixed. Grouped cross-validation did
its job on everything measured inside the training years — the class weighting, the training unit,
the ablation, the tuning null — and was uninformative every time the question was about transfer.
A held-out block of later seasons, rather than grouped folds spread over all years, would have been
the better design and is the main thing to change next time.
