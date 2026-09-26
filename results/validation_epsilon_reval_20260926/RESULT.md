# V23/V36 Validation-Side Epsilon Decision

Status: `VALIDATION_EPSILON_REEVAL_20260926 = COMPLETE`

Evaluations: `8/8 PASS`

Frozen inference epsilon: `0.0125`

## Decision boundary

The inference epsilon is frozen from `protein_cold_valid` only. No test metric
was used to choose it. The existing seed-42/43 selected checkpoints were held
fixed; this experiment did not train a model or reselect a checkpoint.

Source files:

- `results/validation_epsilon_reval_20260926/metrics.json`
- `results/validation_epsilon_reval_20260926/metrics.tsv`

## Two-seed ensemble

| Epsilon | Model | AUPR | AUROC | Log-loss |
|---:|:---|---:|---:|---:|
| 0.025 | V23 | 0.656854 | 0.751216 | 0.683420 |
| 0.025 | V36 | 0.660208 | 0.753149 | 0.692307 |
| 0.0125 | V23 | 0.658000 | 0.752700 | 0.668196 |
| 0.0125 | V36 | 0.660536 | 0.754282 | 0.671459 |

V36 minus V23 at matched epsilon:

| Epsilon | Delta AUPR | Delta AUROC | Delta log-loss |
|---:|---:|---:|---:|
| 0.025 | +0.003354 | +0.001933 | +0.008887 |
| 0.0125 | +0.002536 | +0.001582 | +0.003263 |

V36 has a validation ranking improvement at both epsilons. Its validation
log-loss is worse at both epsilons, including the frozen setting. Therefore the
current evidence supports a ranking claim (AUPR/AUROC), not a claim that V36
provides stable calibration improvement over V23.

## Epsilon effect

Change is `0.0125 - 0.025`; negative log-loss is beneficial.

| Level | Model | Delta AUPR | Delta AUROC | Delta log-loss |
|:---|:---|---:|---:|---:|
| seed 42 | V23 | +0.002389 | +0.002558 | -0.020956 |
| seed 43 | V23 | +0.002261 | +0.002390 | -0.023293 |
| seed 42 | V36 | +0.002479 | +0.005273 | -0.041091 |
| seed 43 | V36 | +0.000083 | +0.000911 | -0.029045 |
| ensemble | V23 | +0.001147 | +0.001483 | -0.015225 |
| ensemble | V36 | +0.000329 | +0.001132 | -0.020848 |

Reducing epsilon to `0.0125` improves all three metrics for every validation
seed and for both two-seed ensembles. This consistent validation-only evidence
is the basis for freezing inference epsilon at `0.0125`.

## Seed-level V36 minus V23

| Epsilon | Seed | Delta AUPR | Delta AUROC | Delta log-loss |
|---:|---:|---:|---:|---:|
| 0.025 | 42 | +0.003348 | -0.002456 | +0.028548 |
| 0.025 | 43 | +0.005997 | +0.004112 | +0.007978 |
| 0.0125 | 42 | +0.003438 | +0.000258 | +0.008413 |
| 0.0125 | 43 | +0.003819 | +0.002633 | +0.002225 |

At the frozen epsilon, both seeds agree in the direction of AUPR and AUROC
improvement, while both show worse V36 log-loss. At epsilon `0.025`, seed 42
does not support an AUROC improvement; this is another reason to quantify
protein-level uncertainty before promoting V36 as the next mainline model.

## Next minimal analysis

Use only the existing V23/V36 seed-42/43 test score arrays at frozen epsilon
`0.0125`. Perform a paired protein-cluster bootstrap and report per-protein
macro metrics, improved/tied/declined counts, count-quartile strata, and the
top-100 most frequent proteins versus the remainder. Do not run model forward,
train, alter a split, or select a checkpoint from test results.
