#!/usr/bin/env python3
"""Paired protein-cluster bootstrap and per-protein V23/V36 error analysis."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from external_benchmark_sepsis import MatrixSplit


MODELS = ("v23", "v36")
SEEDS = (42, 43)
EPSILON = 0.0125
METRICS = ("aupr", "auroc", "log_loss")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, weights=None) -> dict:
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    return {
        "aupr": float(average_precision_score(y_true, scores, sample_weight=weights)),
        "auroc": float(roc_auc_score(y_true, scores, sample_weight=weights)),
        "log_loss": float(
            log_loss(y_true, clipped, sample_weight=weights, labels=[0, 1])
        ),
    }


def delta_metrics(v23: dict, v36: dict) -> dict:
    return {metric: float(v36[metric] - v23[metric]) for metric in METRICS}


def confidence_summary(values: np.ndarray, higher_is_better: bool) -> dict:
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap summary requires non-empty finite values")
    summary = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "probability_positive": float(np.mean(values > 0.0)),
    }
    summary["probability_beneficial"] = float(
        np.mean(values > 0.0) if higher_is_better else np.mean(values < 0.0)
    )
    return summary


def direction_counts(
    values: np.ndarray, tolerance: float, higher_is_better: bool = True
) -> dict:
    beneficial = values if higher_is_better else -values
    return {
        "improved": int(np.sum(beneficial > tolerance)),
        "tied": int(np.sum(np.abs(beneficial) <= tolerance)),
        "declined": int(np.sum(beneficial < -tolerance)),
        "tie_tolerance": tolerance,
        "improvement_direction": "higher" if higher_is_better else "lower",
    }


def load_predictions(args: argparse.Namespace, expected: dict) -> tuple[dict, dict, dict]:
    expected_runs = {
        (item["model"], int(item["seed"])): item
        for item in expected["runs"]
        if float(item["inference_epsilon"]) == EPSILON
    }
    scores = {}
    member_scores = {}
    provenance = {}
    for model in MODELS:
        model_scores = []
        for seed in SEEDS:
            run_dir = (
                Path(args.evaluation_root)
                / model
                / "seed{}".format(seed)
                / "epsilon_0p0125"
            )
            completed_path = run_dir / "completed.json"
            with completed_path.open(encoding="utf-8") as handle:
                completed = json.load(handle)
            if completed.get("model_variant") != {
                "v23": "energy_v23",
                "v36": "mutual_qk_norm_v36",
            }[model]:
                raise ValueError("Unexpected model variant: {}".format(completed_path))
            if int(completed.get("seed")) != seed:
                raise ValueError("Unexpected seed: {}".format(completed_path))
            if float(completed.get("inference_epsilon")) != EPSILON:
                raise ValueError("Unexpected epsilon: {}".format(completed_path))
            if completed.get("test_split") != "protein_cold_test":
                raise ValueError("Unexpected split: {}".format(completed_path))
            if completed.get("test_csv_sha256") != expected["test_csv_sha256"]:
                raise ValueError("Test CSV hash mismatch: {}".format(completed_path))
            expected_run = expected_runs[(model, seed)]
            checkpoint = completed["candidate_checkpoint"]
            if checkpoint["sha256"] != expected_run["candidate_checkpoint_sha256"]:
                raise ValueError("Checkpoint SHA mismatch: {}".format(completed_path))
            if int(checkpoint["best_epoch"]) != int(
                expected_run["candidate_best_epoch"]
            ):
                raise ValueError("Checkpoint epoch mismatch: {}".format(completed_path))
            score_path = run_dir / completed["candidate_score_file"]
            score = np.asarray(np.load(score_path, allow_pickle=False))
            if score.shape != (int(expected["test_rows"]),):
                raise ValueError("Prediction shape mismatch: {}".format(score_path))
            if not np.isfinite(score).all():
                raise ValueError("Non-finite prediction: {}".format(score_path))
            model_scores.append(score)
            member_scores[(model, seed)] = score
            provenance["{}_seed{}".format(model, seed)] = {
                "completed": str(completed_path.resolve()),
                "scores": str(score_path.resolve()),
                "checkpoint_sha256": checkpoint["sha256"],
                "best_epoch": int(checkpoint["best_epoch"]),
            }
        scores[model] = np.mean(np.stack(model_scores, axis=0), axis=0)
    return scores, member_scores, provenance


def verify_recomputed_metrics(
    y_true: np.ndarray,
    scores: dict,
    member_scores: dict,
    expected: dict,
    tolerance: float,
) -> None:
    expected_runs = {
        (item["model"], int(item["seed"])): item["metrics"]
        for item in expected["runs"]
        if float(item["inference_epsilon"]) == EPSILON
    }
    expected_ensembles = {
        item["model"]: item["metrics"]
        for item in expected["ensembles"]
        if float(item["inference_epsilon"]) == EPSILON
    }
    for model in MODELS:
        for seed in SEEDS:
            recomputed = binary_metrics(y_true, member_scores[(model, seed)])
            for metric in METRICS:
                if not np.isclose(
                    recomputed[metric],
                    expected_runs[(model, seed)][metric],
                    rtol=0.0,
                    atol=tolerance,
                ):
                    raise ValueError(
                        "Recomputed {} differs for {} seed{}".format(
                            metric, model, seed
                        )
                    )
        recomputed = binary_metrics(y_true, scores[model])
        for metric in METRICS:
            if not np.isclose(
                recomputed[metric],
                expected_ensembles[model][metric],
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError(
                    "Recomputed ensemble {} differs for {}".format(metric, model)
                )


def prepare_score_order(y_true: np.ndarray, scores: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    group_ends = np.r_[
        np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]), len(scores) - 1
    ]
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    row_loss = -(y_true * np.log(clipped) + (1 - y_true) * np.log1p(-clipped))
    return {"order": order, "group_ends": group_ends, "row_loss": row_loss}


def weighted_metrics_preordered(
    y_true: np.ndarray, weights: np.ndarray, prepared: dict
) -> dict:
    total_weight = float(weights.sum())
    positive_weight = float(np.dot(weights, y_true))
    negative_weight = total_weight - positive_weight
    if positive_weight <= 0.0 or negative_weight <= 0.0:
        raise ValueError("Weighted bootstrap sample does not contain both classes")
    order = prepared["order"]
    ordered_weights = weights[order]
    ordered_labels = y_true[order]
    true_positives = np.cumsum(ordered_weights * ordered_labels)[
        prepared["group_ends"]
    ]
    false_positives = np.cumsum(ordered_weights * (1 - ordered_labels))[
        prepared["group_ends"]
    ]
    recall = true_positives / positive_weight
    predicted_positive_weight = true_positives + false_positives
    precision = np.divide(
        true_positives,
        predicted_positive_weight,
        out=np.ones_like(true_positives, dtype=np.float64),
        where=predicted_positive_weight > 0.0,
    )
    aupr = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    true_positive_rate = np.r_[0.0, recall]
    false_positive_rate = np.r_[0.0, false_positives / negative_weight]
    auroc = np.trapz(true_positive_rate, false_positive_rate)
    weighted_log_loss = np.dot(weights, prepared["row_loss"]) / total_weight
    return {
        "aupr": float(aupr),
        "auroc": float(auroc),
        "log_loss": float(weighted_log_loss),
    }


def verify_reference_macro(
    reference_path: Path,
    macro: dict,
    macro_bootstrap_summary: dict,
    tolerance: float,
) -> dict:
    with reference_path.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    comparisons = {}
    maximum_absolute_difference = 0.0
    for metric in ("aupr", "auroc"):
        for key in ("v23", "v36", "delta"):
            label = "observed_macro.{}.{}".format(metric, key)
            actual = float(macro[metric][key])
            previous = float(reference["macro_both_class_proteins"][metric][key])
            difference = abs(actual - previous)
            comparisons[label] = difference
            maximum_absolute_difference = max(maximum_absolute_difference, difference)
        for key in (
            "mean",
            "median",
            "ci95_low",
            "ci95_high",
            "probability_positive",
        ):
            label = "bootstrap_macro.{}.{}".format(metric, key)
            actual = float(macro_bootstrap_summary[metric][key])
            previous = float(
                reference["bootstrap"]["macro_delta_v36_minus_v23"][metric][key]
            )
            difference = abs(actual - previous)
            comparisons[label] = difference
            maximum_absolute_difference = max(maximum_absolute_difference, difference)
    if maximum_absolute_difference > tolerance:
        raise ValueError(
            "Macro regression check failed: max absolute difference {} > {}".format(
                maximum_absolute_difference, tolerance
            )
        )
    return {
        "reference_metrics": str(reference_path.resolve()),
        "tolerance": tolerance,
        "maximum_absolute_difference": maximum_absolute_difference,
        "comparisons": comparisons,
        "status": "PASS",
    }


def write_metrics_tsv(path: Path, result: dict) -> None:
    rows = []
    for model, metrics in result["observed_micro"].items():
        for metric, value in metrics.items():
            rows.append(("observed_micro", model, metric, "value", value))
    for metric, value in result["observed_micro_delta_v36_minus_v23"].items():
        rows.append(("observed_micro_delta", "v36_minus_v23", metric, "value", value))
    for metric, summary in result["bootstrap"][
        "micro_delta_v36_minus_v23"
    ].items():
        for statistic, value in summary.items():
            rows.append(("bootstrap_micro_delta", "v36_minus_v23", metric, statistic, value))
    for metric, summary in result["macro_both_class_proteins"].items():
        for statistic in ("v23", "v36", "delta"):
            rows.append(("observed_macro", statistic, metric, "value", summary[statistic]))
    for metric, summary in result["bootstrap"][
        "macro_delta_v36_minus_v23"
    ].items():
        for statistic, value in summary.items():
            rows.append(("bootstrap_macro_delta", "v36_minus_v23", metric, statistic, value))
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write("scope\tmodel_or_contrast\tmetric\tstatistic\tvalue\n")
        for row in rows:
            handle.write("{}\t{}\t{}\t{}\t{:.17g}\n".format(*row))


def summarize_stratum(
    name: str,
    protein_ids: np.ndarray,
    codes: np.ndarray,
    y_true: np.ndarray,
    scores: dict,
    per_protein: pd.DataFrame,
    tolerance: float,
) -> dict:
    row_mask = np.isin(codes, protein_ids)
    subset = per_protein.loc[per_protein["protein_index"].isin(protein_ids)]
    eligible = subset.loc[subset["both_classes"]]
    v23_micro = binary_metrics(y_true[row_mask], scores["v23"][row_mask])
    v36_micro = binary_metrics(y_true[row_mask], scores["v36"][row_mask])
    result = {
        "stratum": name,
        "proteins": int(len(protein_ids)),
        "rows": int(row_mask.sum()),
        "row_fraction": float(row_mask.mean()),
        "min_rows_per_protein": int(subset["rows"].min()),
        "max_rows_per_protein": int(subset["rows"].max()),
        "both_class_proteins": int(len(eligible)),
        "micro_v23": v23_micro,
        "micro_v36": v36_micro,
        "micro_delta_v36_minus_v23": delta_metrics(v23_micro, v36_micro),
    }
    if len(eligible):
        for metric in METRICS:
            if metric != "log_loss":
                metric_rows = eligible
            else:
                metric_rows = subset
            delta = metric_rows["delta_{}".format(metric)].to_numpy()
            result["macro_{}_v23".format(metric)] = float(
                metric_rows["v23_{}".format(metric)].mean()
            )
            result["macro_{}_v36".format(metric)] = float(
                metric_rows["v36_{}".format(metric)].mean()
            )
            result["macro_delta_{}".format(metric)] = float(np.mean(delta))
            result["direction_{}".format(metric)] = direction_counts(
                delta, tolerance, higher_is_better=(metric != "log_loss")
            )
    return result


def main(args: argparse.Namespace) -> None:
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    if not 0 <= args.sklearn_audit_replicates <= args.bootstrap_replicates:
        raise ValueError(
            "--sklearn-audit-replicates must be between zero and all replicates"
        )
    if (
        args.tie_tolerance < 0.0
        or args.metric_tolerance < 0.0
        or args.regression_tolerance < 0.0
    ):
        raise ValueError("Tolerances must be non-negative")
    expected_path = Path(args.expected_metrics).resolve()
    with expected_path.open(encoding="utf-8") as handle:
        expected = json.load(handle)
    if expected.get("test_split") != "protein_cold_test":
        raise ValueError("Expected metrics are not protein_cold_test")
    if list(expected.get("seeds", [])) != list(SEEDS):
        raise ValueError("Expected metrics do not use seeds 42/43")

    test_csv = Path(args.test_csv).resolve()
    csv_sha256 = file_sha256(test_csv)
    if csv_sha256 != expected["test_csv_sha256"]:
        raise ValueError("Test CSV SHA-256 differs from the locked evaluation")
    protein_frame = pd.read_csv(test_csv, usecols=[args.protein_column])
    if protein_frame[args.protein_column].isna().any():
        raise ValueError("Protein identity column contains missing values")
    proteins = protein_frame[args.protein_column].astype(str).to_numpy()
    if len(proteins) != int(expected["test_rows"]):
        raise ValueError("Protein identity rows do not align with predictions")

    test = MatrixSplit("protein_cold_test", preload=True)
    y_true = np.asarray(test.y[: len(test)], dtype=np.uint8)
    if len(y_true) != len(proteins):
        raise ValueError("Matrix labels do not align with test CSV")
    scores, member_scores, provenance = load_predictions(args, expected)
    verify_recomputed_metrics(
        y_true, scores, member_scores, expected, args.metric_tolerance
    )

    unique_proteins, codes = np.unique(proteins, return_inverse=True)
    counts = np.bincount(codes, minlength=len(unique_proteins))
    audit = {
        "test_csv": str(test_csv),
        "test_csv_sha256": csv_sha256,
        "rows": len(y_true),
        "positive_rows": int(y_true.sum()),
        "proteins": int(len(unique_proteins)),
        "protein_column": args.protein_column,
        "epsilon": EPSILON,
        "prediction_provenance": provenance,
        "metrics_recomputed_match_summary": True,
    }
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_INPUT_AUDIT_PASS", **audit}, indent=2))
        return

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for protein_index, protein in enumerate(unique_proteins):
        row_mask = codes == protein_index
        labels = y_true[row_mask]
        both_classes = bool(labels.min() != labels.max())
        record = {
            "protein_index": protein_index,
            "protein_sha256": hashlib.sha256(protein.encode("utf-8")).hexdigest(),
            "protein_length": len(protein),
            "rows": int(row_mask.sum()),
            "positives": int(labels.sum()),
            "positive_rate": float(labels.mean()),
            "both_classes": both_classes,
        }
        for model in MODELS:
            model_scores = scores[model][row_mask]
            record["{}_log_loss".format(model)] = float(
                log_loss(
                    labels,
                    np.clip(model_scores, 1e-7, 1.0 - 1e-7),
                    labels=[0, 1],
                )
            )
            if both_classes:
                record["{}_aupr".format(model)] = float(
                    average_precision_score(labels, model_scores)
                )
                record["{}_auroc".format(model)] = float(
                    roc_auc_score(labels, model_scores)
                )
            else:
                record["{}_aupr".format(model)] = np.nan
                record["{}_auroc".format(model)] = np.nan
        for metric in METRICS:
            record["delta_{}".format(metric)] = (
                record["v36_{}".format(metric)]
                - record["v23_{}".format(metric)]
            )
        records.append(record)
    per_protein = pd.DataFrame.from_records(records)
    eligible = per_protein.loc[per_protein["both_classes"]].copy()

    observed = {
        model: binary_metrics(y_true, scores[model]) for model in MODELS
    }
    observed_delta = delta_metrics(observed["v23"], observed["v36"])
    macro = {}
    directions = {}
    for metric in ("aupr", "auroc"):
        macro[metric] = {
            "v23": float(eligible["v23_{}".format(metric)].mean()),
            "v36": float(eligible["v36_{}".format(metric)].mean()),
            "delta": float(eligible["delta_{}".format(metric)].mean()),
            "eligible_proteins": int(len(eligible)),
        }
        directions[metric] = direction_counts(
            eligible["delta_{}".format(metric)].to_numpy(), args.tie_tolerance
        )
    macro["log_loss"] = {
        "v23": float(per_protein["v23_log_loss"].mean()),
        "v36": float(per_protein["v36_log_loss"].mean()),
        "delta": float(per_protein["delta_log_loss"].mean()),
        "eligible_proteins": int(len(per_protein)),
    }
    directions["log_loss"] = direction_counts(
        per_protein["delta_log_loss"].to_numpy(),
        args.tie_tolerance,
        higher_is_better=False,
    )

    count_order = np.argsort(counts, kind="stable")
    strata = []
    for index, protein_ids in enumerate(np.array_split(count_order, 4), start=1):
        strata.append(
            summarize_stratum(
                "count_quartile_{}".format(index),
                protein_ids,
                codes,
                y_true,
                scores,
                per_protein,
                args.tie_tolerance,
            )
        )
    high_frequency = np.argsort(-counts, kind="stable")[: min(100, len(counts))]
    remaining = np.setdiff1d(np.arange(len(counts)), high_frequency)
    strata.append(
        summarize_stratum(
            "top_100_by_row_count",
            high_frequency,
            codes,
            y_true,
            scores,
            per_protein,
            args.tie_tolerance,
        )
    )
    strata.append(
        summarize_stratum(
            "all_except_top_100",
            remaining,
            codes,
            y_true,
            scores,
            per_protein,
            args.tie_tolerance,
        )
    )

    rng = np.random.default_rng(args.bootstrap_seed)
    prepared_scores = {
        model: prepare_score_order(y_true, scores[model]) for model in MODELS
    }
    micro_bootstrap = {metric: [] for metric in METRICS}
    macro_bootstrap = {metric: [] for metric in ("aupr", "auroc")}
    eligible_deltas = {
        metric: eligible["delta_{}".format(metric)].to_numpy()
        for metric in ("aupr", "auroc")
    }
    eligible_protein_ids = eligible["protein_index"].to_numpy(dtype=np.int64)
    for replicate in range(args.bootstrap_replicates):
        sampled = rng.integers(0, len(unique_proteins), size=len(unique_proteins))
        multiplicity = np.bincount(sampled, minlength=len(unique_proteins))
        weights = multiplicity[codes]
        if np.sum(weights[y_true == 0]) == 0 or np.sum(weights[y_true == 1]) == 0:
            raise RuntimeError(
                "Bootstrap replicate {} does not contain both classes".format(
                    replicate
                )
            )
        v23_metrics = weighted_metrics_preordered(
            y_true, weights, prepared_scores["v23"]
        )
        v36_metrics = weighted_metrics_preordered(
            y_true, weights, prepared_scores["v36"]
        )
        if replicate < args.sklearn_audit_replicates:
            for model, fast_metrics in (
                ("v23", v23_metrics),
                ("v36", v36_metrics),
            ):
                sklearn_metrics = binary_metrics(y_true, scores[model], weights)
                for metric in METRICS:
                    if not np.isclose(
                        fast_metrics[metric],
                        sklearn_metrics[metric],
                        rtol=0.0,
                        atol=args.metric_tolerance,
                    ):
                        raise ValueError(
                            "Preordered {} differs from sklearn for replicate {} {}".format(
                                metric, replicate, model
                            )
                        )
        for metric in METRICS:
            micro_bootstrap[metric].append(
                v36_metrics[metric] - v23_metrics[metric]
            )
        macro_weights = multiplicity[eligible_protein_ids]
        macro_weight_sum = int(macro_weights.sum())
        if macro_weight_sum == 0:
            raise RuntimeError(
                "Bootstrap replicate {} contains no both-class protein".format(
                    replicate
                )
            )
        for metric in ("aupr", "auroc"):
            macro_bootstrap[metric].append(
                float(
                    np.dot(eligible_deltas[metric], macro_weights)
                    / macro_weight_sum
                )
            )
        if args.progress_every and (replicate + 1) % args.progress_every == 0:
            print("bootstrap {}/{}".format(replicate + 1, args.bootstrap_replicates))

    replicate_validation = {}
    for scope, values_by_metric in (
        ("micro", micro_bootstrap),
        ("macro", macro_bootstrap),
    ):
        replicate_validation[scope] = {}
        for metric, values in values_by_metric.items():
            array = np.asarray(values)
            finite_count = int(np.isfinite(array).sum())
            if len(array) != args.bootstrap_replicates or finite_count != len(array):
                raise RuntimeError(
                    "{} {} bootstrap is incomplete or non-finite: {}/{}".format(
                        scope, metric, finite_count, args.bootstrap_replicates
                    )
                )
            replicate_validation[scope][metric] = {
                "replicates": len(array),
                "finite": finite_count,
                "status": "PASS",
            }

    macro_bootstrap_summary = {
        metric: confidence_summary(np.asarray(values), higher_is_better=True)
        for metric, values in macro_bootstrap.items()
    }
    regression_checks = {
        "sklearn_equivalence": {
            "replicates_checked": args.sklearn_audit_replicates,
            "metrics": list(METRICS),
            "tolerance": args.metric_tolerance,
            "status": "PASS",
        }
    }
    if args.reference_metrics:
        regression_checks["previous_macro_results"] = verify_reference_macro(
            Path(args.reference_metrics).resolve(),
            macro,
            macro_bootstrap_summary,
            args.regression_tolerance,
        )

    bootstrap = {
        "method": "paired protein-cluster bootstrap with replacement",
        "average_precision_implementation": (
            "score-preordered weighted average precision with safe zero-denominator "
            "handling; audited against sklearn.metrics.average_precision_score"
        ),
        "replicates_requested": args.bootstrap_replicates,
        "replicates_completed": len(micro_bootstrap["aupr"]),
        "seed": args.bootstrap_seed,
        "replicate_validation": replicate_validation,
        "micro_delta_v36_minus_v23": {
            metric: confidence_summary(
                np.asarray(values), higher_is_better=(metric != "log_loss")
            )
            for metric, values in micro_bootstrap.items()
        },
        "macro_delta_v36_minus_v23": macro_bootstrap_summary,
    }

    result = {
        "analysis": "v23_v36_protein_paired_bootstrap_v2_safe_ap",
        "input_audit": audit,
        "observed_micro": observed,
        "observed_micro_delta_v36_minus_v23": observed_delta,
        "macro_both_class_proteins": macro,
        "protein_direction_counts": directions,
        "strata": strata,
        "bootstrap": bootstrap,
        "regression_checks": regression_checks,
        "interpretation_constraints": [
            "Uses frozen inference epsilon 0.0125 selected on protein_cold_valid.",
            "Uses existing seed42/43 test predictions; no model inference was run.",
            "Test analysis remains exploratory and does not select checkpoints.",
        ],
    }
    with (output_dir / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    write_metrics_tsv(output_dir / "metrics.tsv", result)
    per_protein.to_csv(output_dir / "per_protein_metrics.csv", index=False)
    pd.json_normalize(strata).to_csv(output_dir / "strata_metrics.csv", index=False)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--expected-metrics", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--protein-column", default="Protein")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260926)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--metric-tolerance", type=float, default=1e-8)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--sklearn-audit-replicates", type=int, default=5)
    parser.add_argument("--reference-metrics")
    parser.add_argument("--regression-tolerance", type=float, default=1e-12)
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
