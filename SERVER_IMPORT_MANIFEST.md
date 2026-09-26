# GloBIS-DTI Server Import Manifest

## Scope

This manifest defines the minimum small-file import needed to prepare the next
GloBIS-DTI experiment through GitHub. The authoritative server workspace is:

`/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM_Sepsis_Dev`

The immediate experiment is an evaluation-only comparison of the existing V23
and V36 checkpoints with matched seeds, selected checkpoints, validation/test
inputs, and ensemble membership under `epsilon=0.025`, `0.0125`, and, only if
needed, `0`. It must write to new output directories and must not overwrite any
existing experiment artifact.

## Import now: source and launch definitions

Copy these repository-relative files from the server workspace without editing
them:

- `code/evaluate_atom_segment_test.py`
- `code/run_full_two_seed_mutual_qk_norm_v36.sh`
- `code/external_benchmark_atom_segment_energy_v23.py`
- `code/external_benchmark_atom_segment_mutual_v30.py`
- `code/external_benchmark_atom_segment_qk_norm_v26.py`
- `code/external_benchmark_atom_segment_mutual_qk_norm_v36.py`

The four architecture files are required to preserve the existing definitions:
V23 is the fixed base, V30 adds bidirectional mutual-attention context, V26 adds
final routing Q/K L2 normalization, and V36 combines the V30 and V26 changes.

## Import now: authoritative documentation and small audit metadata

- `GLOBIS_DTI_HANDOFF_2026-09-25.md`
- `external_benchmark/audits/data_training_20260925/REPORT_CN.md`
- `external_benchmark/audits/data_training_20260925/data_summary.json`
- `external_benchmark/audits/data_training_20260925/signal_metrics.json`
- `external_benchmark/audits/data_training_20260925/training_summary.json`
- `external_benchmark/audits/data_training_20260925/matched_configs.json`
- `paper_evidence/V23_DATA_SOURCE_AUDIT_2026-09-05/README_CN.md`
- `paper_evidence/V23_DATA_SOURCE_AUDIT_2026-09-05/verification_summary.json`
- `MODEL_MODIFICATION_HANDOFF_2026-09-08.md`
- `V23_PAPER_WRITING_HANDOFF_2026-09-05.md`
- `HANDOFF_2026-08-21.md`

Do not import `code/audit_data_training_20260925.py`; the handoff identifies it
as an obsolete 77-byte stub. The reproducible audit implementation, when later
needed, is `code/audit_globis_data_training_20260925.py`.

## Import now: run metadata only

From the exact V23 and V36 seed-42/seed-43 runs referenced by
`matched_configs.json` and `code/run_full_two_seed_mutual_qk_norm_v36.sh`, copy
only the following small metadata files while preserving their repository-
relative directory structure:

- `run_config.json`
- `completed.json`

If checkpoint hashes are not already present in those JSON files, add one small
`CHECKPOINT_MANIFEST.json` under a new audit directory. It may contain only:

- model version;
- seed;
- selected epoch;
- server-local checkpoint path;
- checkpoint SHA-256;
- architecture entry point;
- evaluation split identifier;
- cache identifiers or server-local paths;
- ensemble membership.

The checkpoint files themselves must remain on the GPU server.

## Explicitly keep on the server

Do not add any of the following to GitHub:

- `*.pt`, `*.pth`, `*.ckpt`, model weights, optimizer states, or checkpoints;
- raw or processed datasets, including the matrix directory and protein-cold
  `{train,valid,test}.csv` files;
- feature caches, embedding caches, ESM2/BerMol caches, or downloaded structures;
- full prediction arrays or complete experiment-output directories;
- large logs, temporary files, package caches, Jupyter checkpoints, or secrets;
- credentials, tokens, cookies, private keys, environment files, or passwords.

In particular, keep these server-side paths out of Git:

- `/mnt/home/dachuang/dtiam/DTIAM-main/DTIAM-main/external_benchmark/matrices`
- `/mnt/home/dachuang/dti_in_domain_test/splits/protein_cold_start/`
- checkpoint and cache paths referenced by the run script or run metadata.

## Import verification

Before committing the import on the server:

1. Confirm the repository branch and a clean starting status.
2. Confirm every imported path is a small source, configuration, JSON, Markdown,
   or shell-script file.
3. Inspect `git diff --stat` and the staged file list.
4. Verify no dataset, checkpoint, cache, full prediction output, or secret is
   staged.
5. Commit and push the import on a non-`main`/non-`master` collaboration branch.

After the import reaches GitHub, the local development side should first review
the evaluator CLI and existing launcher, then prepare a new evaluation-only run
script. No model redesign or training should begin before the matched epsilon
evaluation is completed and returned as small result summaries.
