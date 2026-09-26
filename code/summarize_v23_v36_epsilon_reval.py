#!/usr/bin/env python3
"""Summarize matched V23/V36 epsilon re-evaluations without saving predictions."""

import argparse
import json
import math
from pathlib import Path


MODELS = {
    "v23": "energy_v23",
    "v36": "mutual_qk_norm_v36",
}
SEEDS = (42, 43)
EPSILONS = (0.025, 0.0125)
REQUIRED_METRICS = ("aupr", "auroc", "log_loss")


def epsilon_directory(epsilon: float) -> str:
    return "epsilon_{}".format(format(epsilon, "g").replace(".", "p"))


def scalar_metrics(payload: dict) -> dict:
    metrics = payload.get("candidate_metrics", {})
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise KeyError("candidate_metrics missing: {}".format(", ".join(missing)))
    return {key: float(metrics[key]) for key in REQUIRED_METRICS}


def checkpoint_sha256(metadata: dict) -> str:
    for key in ("sha256", "checkpoint_sha256", "file_sha256"):
        value = metadata.get(key)
        if value:
            return str(value)
    raise KeyError("checkpoint metadata does not contain a SHA-256")


def assert_close(label: str, actual: float, expected: float, tolerance: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError("{}={} expected {}".format(label, actual, expected))


def main(args: argparse.Namespace) -> None:
    import numpy as np

    from external_benchmark_sepsis import MatrixSplit, binary_metrics

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Evaluation root does not exist: {}".format(root))

    output_json = Path(args.output_json or root / "metrics.json").resolve()
    output_tsv = Path(args.output_tsv or root / "metrics.tsv").resolve()
    for output in (output_json, output_tsv):
        if output.exists():
            raise FileExistsError("Refusing to overwrite summary: {}".format(output))

    test = MatrixSplit(args.test_split, preload=True)
    y_true = np.asarray(test.y[: len(test)], dtype=np.uint8)
    common_test_hash = None
    common_rows = None
    checkpoint_sha_by_model_seed = {}
    base_sha_by_seed = {}
    scores = {}
    runs = []

    for model, expected_variant in MODELS.items():
        for seed in SEEDS:
            for epsilon in EPSILONS:
                run_dir = root / model / "seed{}".format(seed) / epsilon_directory(epsilon)
                completed_path = run_dir / "completed.json"
                with completed_path.open(encoding="utf-8") as handle:
                    completed = json.load(handle)

                if completed.get("model_variant") != expected_variant:
                    raise ValueError("Unexpected model variant in {}".format(completed_path))
                if int(completed.get("seed")) != seed:
                    raise ValueError("Unexpected seed in {}".format(completed_path))
                assert_close(
                    "trained_epsilon", float(completed.get("trained_epsilon")), 0.025, 1e-12
                )
                assert_close(
                    "inference_epsilon",
                    float(completed.get("inference_epsilon")),
                    epsilon,
                    1e-12,
                )
                if completed.get("test_split") != args.test_split:
                    raise ValueError("Unexpected test split in {}".format(completed_path))

                rows = int(completed.get("rows"))
                test_hash = completed.get("test_csv_sha256")
                if rows != len(y_true):
                    raise ValueError("Test row count mismatch in {}".format(completed_path))
                if common_rows is None:
                    common_rows = rows
                    common_test_hash = test_hash
                elif rows != common_rows or test_hash != common_test_hash:
                    raise ValueError("Evaluations do not use the same test data")

                candidate_sha = completed["candidate_checkpoint"]["sha256"]
                checkpoint_key = (model, seed)
                previous_candidate_sha = checkpoint_sha_by_model_seed.setdefault(
                    checkpoint_key, candidate_sha
                )
                if candidate_sha != previous_candidate_sha:
                    raise ValueError("Checkpoint changed across epsilon for {}".format(checkpoint_key))

                base_sha = checkpoint_sha256(completed["base_checkpoint"])
                previous_base_sha = base_sha_by_seed.setdefault(seed, base_sha)
                if base_sha != previous_base_sha:
                    raise ValueError("V23/V36 base checkpoint differs for seed {}".format(seed))

                score_path = run_dir / completed["candidate_score_file"]
                score = np.asarray(np.load(score_path, allow_pickle=False))
                if score.shape != (rows,) or not np.isfinite(score).all():
                    raise ValueError("Invalid prediction array: {}".format(score_path))
                scores[(model, seed, epsilon)] = score

                saved_metrics = scalar_metrics(completed)
                recomputed_metrics = binary_metrics(y_true, score)
                for metric in REQUIRED_METRICS:
                    assert_close(
                        "{} {}".format(completed_path, metric),
                        float(recomputed_metrics[metric]),
                        saved_metrics[metric],
                        args.metric_tolerance,
                    )
                runs.append(
                    {
                        "model": model,
                        "seed": seed,
                        "inference_epsilon": epsilon,
                        "metrics": saved_metrics,
                        "candidate_checkpoint_sha256": candidate_sha,
                        "candidate_best_epoch": int(
                            completed["candidate_checkpoint"]["best_epoch"]
                        ),
                    }
                )

    ensembles = []
    for model in MODELS:
        for epsilon in EPSILONS:
            ensemble_scores = np.mean(
                np.stack([scores[(model, seed, epsilon)] for seed in SEEDS], axis=0),
                axis=0,
            )
            metrics = binary_metrics(y_true, ensemble_scores)
            ensembles.append(
                {
                    "model": model,
                    "seeds": list(SEEDS),
                    "inference_epsilon": epsilon,
                    "metrics": {
                        key: float(metrics[key]) for key in REQUIRED_METRICS
                    },
                }
            )

    summary = {
        "evaluation": "v23_v36_epsilon_reval_20260926",
        "test_split": args.test_split,
        "test_rows": common_rows,
        "test_csv_sha256": common_test_hash,
        "trained_epsilon": 0.025,
        "inference_epsilons": list(EPSILONS),
        "seeds": list(SEEDS),
        "runs": runs,
        "ensembles": ensembles,
        "selection_note": "Existing selected checkpoints; no test-based reselection.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with output_tsv.open("x", encoding="utf-8", newline="") as handle:
        handle.write("level\tmodel\tseed\tinference_epsilon\taupr\tauroc\tlog_loss\n")
        for run in runs:
            metrics = run["metrics"]
            handle.write(
                "seed\t{model}\t{seed}\t{epsilon:.6g}\t{aupr:.12g}\t{auroc:.12g}\t{log_loss:.12g}\n".format(
                    model=run["model"],
                    seed=run["seed"],
                    epsilon=run["inference_epsilon"],
                    **metrics,
                )
            )
        for ensemble in ensembles:
            metrics = ensemble["metrics"]
            handle.write(
                "ensemble2\t{model}\t42,43\t{epsilon:.6g}\t{aupr:.12g}\t{auroc:.12g}\t{log_loss:.12g}\n".format(
                    model=ensemble["model"],
                    epsilon=ensemble["inference_epsilon"],
                    **metrics,
                )
            )

    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--test-split", default="protein_cold_test")
    parser.add_argument("--output-json")
    parser.add_argument("--output-tsv")
    parser.add_argument("--metric-tolerance", type=float, default=1e-8)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
