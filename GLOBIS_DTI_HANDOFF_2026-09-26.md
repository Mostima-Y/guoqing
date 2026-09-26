# GloBIS-DTI Handoff Update (2026-09-26)

This document extends, rather than replaces,
`GLOBIS_DTI_HANDOFF_2026-09-25.md`. Model architectures, checkpoints, benchmark
splits, and historical outputs remain unchanged.

## Frozen inference policy

Inference epsilon is frozen at `0.0125`. The decision was made exclusively from
the completed `protein_cold_valid` comparison in
`results/validation_epsilon_reval_20260926/`. Relative to epsilon `0.025`, the
smaller epsilon improves AUPR, AUROC, and log-loss for V23 and V36 at seed 42,
seed 43, and in the two-seed ensembles. Test metrics were not used for this
choice.

At frozen epsilon, validation ensemble V36 minus V23 is AUPR `+0.002536` and
AUROC `+0.001582`, but log-loss `+0.003263` (worse). Describe the current V36
signal as a ranking improvement only. Do not claim stable calibration gains.

The earlier test-side epsilon comparison remains exploratory sensitivity
analysis. It showed that matched inference epsilon does not explain away the
V36 test ranking gain, but it is not an unbiased epsilon-selection source.

## Next experiment: no training and no inference

The next decision gate is whether the frozen-epsilon V36 ranking gain is broad
across proteins or concentrated in high-frequency proteins. Run:

- `code/run_v23_v36_protein_bootstrap.sh`
- `code/analyze_v23_v36_protein_bootstrap.py`

### Input sufficiency audit

The existing artifacts are sufficient; another model forward pass is not
needed. The validation evaluator saved one score array per epsilon and its
`completed.json` records split, rows, seed, model variant, checkpoint hash and
score filenames. Those validation arrays are no longer needed for the next
test-side error analysis because the epsilon decision is already frozen.

For the test analysis, each of the four matched V23/V36 seed-42/43 directories
at epsilon `0.0125` has a `completed.json` that names its candidate score
array. The committed test summary supplies the locked CSV SHA-256, row count,
checkpoint hashes, selected epochs and reference seed/ensemble metrics. The
test CSV supplies row-aligned protein identity, while `MatrixSplit` supplies
the same labels used by the evaluator. The new dry-run verifies all of these
conditions and recomputes every seed and ensemble metric before accepting the
inputs. Thus the `.npy` scores, row metadata, and exact protein strings provide
the grouping and paired predictions required for the planned analysis.

The analysis consumes existing V23/V36 seed-42/43 score arrays under
`external_benchmark/test_evaluations/epsilon_reval_20260926/`, plus their
`completed.json` files and the locked protein-cold test CSV. It verifies the
test CSV hash, row count, selected checkpoint hashes/epochs, model variants,
seeds, and epsilon before analysis. Protein strings are used only for grouping;
the output records SHA-256 identities rather than raw sequences.

Expected new server-only output directory:

`external_benchmark/audits/protein_bootstrap_20260926/`

It must not already exist. The launcher refuses to overwrite it. Outputs are
small tabular/JSON summaries only:

- `metrics.json`
- `per_protein_metrics.csv`
- `strata_metrics.csv`

The paired bootstrap resamples proteins, not rows. It reports micro AUPR,
AUROC, and log-loss deltas; macro AUPR/AUROC over proteins containing both
classes; per-protein direction counts; protein-count quartiles; and top-100
high-frequency proteins versus all remaining proteins. This directly tests
whether micro gains persist under equal-protein weighting and outside dominant
proteins.

Run `--dry-run` first. Dry-run loads and audits the four existing score arrays
but creates no output and performs no bootstrap. No GPU is required.
The launcher is run from the Git checkout (normally
`/mnt/home/dachuang/guoqing`) and reads dependencies/artifacts from the
authoritative workspace through its default `--root`; no script or prediction
array needs to be copied between the two directories.

## Interpretation boundary

The next analysis uses test predictions already inspected, so it is exploratory
error analysis. It may decide whether V36 merits further validation or new
training, but it must not be used to tune epsilon, reselect checkpoints, modify
the frozen benchmark, or claim an independent generalization estimate.
