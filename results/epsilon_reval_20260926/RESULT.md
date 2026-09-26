# V23/V36 Test-Side Epsilon Re-evaluation

Status: `EPSILON_REEVAL_20260926 = COMPLETE`
Evaluations: `8/8 PASS`

## Scope

Existing V23 and V36 seed-42/43 selected checkpoints were evaluated on the
same protein-cold test set with inference epsilon `0.025` and `0.0125`.
No model was trained, no checkpoint was reselected, and the historical
five-seed V23 ensemble was not included.

Source results:

- `results/epsilon_reval_20260926/metrics.json`
- `results/epsilon_reval_20260926/metrics.tsv`

All eight evaluations use test CSV SHA-256
`1de92afebe892fc512f639735872471231be539f3d7426ee686656acce44ddb7`
and 179,829 rows.

## Two-seed ensemble comparison

| Inference epsilon | V23 AUPR | V36 AUPR | Delta AUPR | V23 AUROC | V36 AUROC | Delta AUROC | Delta log-loss (V36 - V23) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.025 | 0.631572 | 0.638367 | +0.006795 | 0.716678 | 0.723825 | +0.007147 | -0.004165 |
| 0.0125 | 0.632082 | 0.636306 | +0.004224 | 0.719216 | 0.723285 | +0.004070 | -0.002866 |

V36 exceeds V23 in ensemble AUPR and AUROC at both tested inference epsilons.
The V36 advantage is therefore not explained by the historical test inference
epsilon of `0.0125`; its ranking advantage is larger at `0.025`.

## Effect of reducing inference epsilon

Change shown is `epsilon 0.0125 - epsilon 0.025`.

| Model | Delta AUPR | Delta AUROC | Delta log-loss |
|---|---:|---:|---:|
| V23 | +0.000510 | +0.002538 | -0.024250 |
| V36 | -0.002061 | -0.000540 | -0.022950 |

Reducing epsilon substantially improves log-loss for both models. Its ranking
effect is architecture-dependent: V23 improves slightly, while V36 declines
slightly. Epsilon therefore changes the ranking/calibration trade-off rather
than acting as a uniformly beneficial shrinkage parameter.

## Seed consistency

V36-minus-V23 ranking deltas are positive for both seeds at both epsilons:

| Epsilon | Seed | Delta AUPR | Delta AUROC | Delta log-loss |
|---:|---:|---:|---:|---:|
| 0.025 | 42 | +0.005358 | +0.004371 | +0.014219 |
| 0.025 | 43 | +0.003640 | +0.005764 | -0.000835 |
| 0.0125 | 42 | +0.004941 | +0.003696 | +0.000985 |
| 0.0125 | 43 | +0.002492 | +0.003427 | -0.002416 |

The AUPR/AUROC direction is not driven by one seed. Single-seed log-loss is
not uniformly improved: V36 is worse for seed 42 at both epsilons, although the
two-seed ensemble log-loss is better.

## Interpretation boundary

These are exploratory test sensitivity results. They support:

- `V36 > V23` in test AUPR/AUROC at both tested epsilons;
- `EPSILON_ARTIFACT_EXPLAINS_V36_GAIN = NO` for this two-seed comparison;
- smaller epsilon improves probability log-loss for both architectures;
- the ranking response to epsilon differs between V23 and V36.

They do **not** support selecting `0.025` as an unbiased best inference epsilon.
The test set has already been inspected, no protein-level uncertainty interval
is included here, and the project has documented split and benchmark risks.

## Next minimal experiment

Evaluate the same existing V23/V36 seed-42/43 selected checkpoints on the
unchanged `protein_cold_valid` split at inference epsilon `0.025` and `0.0125`.
Do not train or reselect checkpoints. Use existing validation token/segment
caches, write to a new directory, and do not read the test split. The resulting
validation comparison can inform a future epsilon policy without selecting it
from test performance.

`epsilon=0` is intentionally excluded from this round.
