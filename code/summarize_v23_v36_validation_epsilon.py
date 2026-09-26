#!/usr/bin/env python3
"""Summarize matched validation-side epsilon evaluations for V23 and V36."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from external_benchmark_sepsis import MatrixSplit, binary_metrics
from summarize_v23_v36_epsilon_reval import checkpoint_sha256


MODELS = {
    "v23": "energy_v23",
    "v36": "mutual_qk_norm_v36",
}
SEEDS = (42, 43)
EPSILONS = (0.025, 0.0125)
METRICS = ("aupr", "auroc", "log_loss")


def assert_close(label: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("{}={} expected {}".format(label, actual, expected))


def main(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Validation evaluation root missing: {}".format(root))
    output_json = root / "metrics.json"
    output_tsv = root / "metrics.tsv"
    for output in (output_json, output_tsv):
        if output.exists():
            raise FileExistsError("Refusing to overwrite summary: {}".format(output))

    split_name = "protein_cold_valid"
    validation = MatrixSplit(split_name, preload=True)
    y_true = np.asarray(validation.y[: len(validation)], dtype=np.uint8)
    scores = {}
    runs = []
    base_sha_by_seed = {}

    for model, expected_variant in MODELS.items():
        for seed in SEEDS:
            run_dir = root / model / "seed{}".format(seed)
            with (run_dir / "completed.json").open(encoding="utf-8") as handle:
                completed = json.load(handle)
            if completed.get("evaluation_split") != split_name:
                raise ValueError("Unexpected split in {}".format(run_dir))
            if completed.get("model_variant") != expected_variant:
                raise ValueError("Unexpected model in {}".format(run_dir))
            if int(completed.get("seed")) != seed:
                raise ValueError("Unexpected seed in {}".format(run_dir))
            if int(completed.get("rows")) != len(y_true):
                raise ValueError("Validation row mismatch in {}".format(run_dir))
            assert_close("trained_epsilon", float(completed["trained_epsilon"]), 0.025)

            base_sha = checkpoint_sha256(completed["base_checkpoint"])
            previous_base_sha = base_sha_by_seed.setdefault(seed, base_sha)
            if base_sha != previous_base_sha:
                raise ValueError("V23/V36 base differs for seed {}".format(seed))

            for epsilon in EPSILONS:
                key = "{:.6f}".format(epsilon)
                score_path = run_dir / completed["candidate_score_files"][key]
                score = np.asarray(np.load(score_path, allow_pickle=False))
                if score.shape != (len(y_true),) or not np.isfinite(score).all():
                    raise ValueError("Invalid scores: {}".format(score_path))
                metrics = binary_metrics(y_true, score)
                saved = completed["candidate_metrics"][key]
                for metric in METRICS:
                    assert_close(
                        "{} {}".format(score_path, metric),
                        float(metrics[metric]),
                        float(saved[metric]),
                    )
                scores[(model, seed, epsilon)] = score
                runs.append(
                    {
                        "model": model,
                        "seed": seed,
                        "inference_epsilon": epsilon,
                        "metrics": {metric: float(saved[metric]) for metric in METRICS},
                        "candidate_checkpoint_sha256": completed[
                            "candidate_checkpoint"
                        ]["sha256"],
                        "candidate_best_epoch": int(
                            completed["candidate_checkpoint"]["best_epoch"]
                        ),
                    }
                )

    ensembles = []
    for model in MODELS:
        for epsilon in EPSILONS:
            ensemble_scores = np.mean(
                np.stack([scores[(model, seed, epsilon)] for seed in SEEDS]), axis=0
            )
            metrics = binary_metrics(y_true, ensemble_scores)
            ensembles.append(
                {
                    "model": model,
                    "seeds": list(SEEDS),
                    "inference_epsilon": epsilon,
                    "metrics": {metric: float(metrics[metric]) for metric in METRICS},
                }
            )

    summary = {
        "evaluation": "v23_v36_validation_epsilon_reval_20260926",
        "evaluation_split": split_name,
        "rows": len(y_true),
        "trained_epsilon": 0.025,
        "inference_epsilons": list(EPSILONS),
        "seeds": list(SEEDS),
        "runs": runs,
        "ensembles": ensembles,
        "selection_note": "Existing selected checkpoints; no validation reselection.",
        "test_data_read": False,
    }
    with output_json.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with output_tsv.open("x", encoding="utf-8", newline="") as handle:
        handle.write("level\tmodel\tseed\tinference_epsilon\taupr\tauroc\tlog_loss\n")
        for run in runs:
            handle.write(
                "seed\t{model}\t{seed}\t{inference_epsilon:.6g}\t{aupr:.12g}\t{auroc:.12g}\t{log_loss:.12g}\n".format(
                    **run, **run["metrics"]
                )
            )
        for ensemble in ensembles:
            handle.write(
                "ensemble2\t{model}\t42,43\t{inference_epsilon:.6g}\t{aupr:.12g}\t{auroc:.12g}\t{log_loss:.12g}\n".format(
                    **ensemble, **ensemble["metrics"]
                )
            )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    main(parser.parse_args())
