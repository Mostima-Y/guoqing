#!/usr/bin/env python3
"""Evaluate one locked atom-segment checkpoint on an explicitly unlocked test."""

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import external_benchmark_atom_segment as atom_segment_base
from analyze_atom_segment_residual_scaling import predict_epsilon_grid
from build_drug_substructure_cache import (
    build_token_arrays,
    file_sha256,
    load_bermol_artifacts,
    read_ligands,
)
from external_benchmark_atom_segment import (
    AtomSegmentInteractionNet,
    DrugTokenLookup,
    load_base_checkpoint,
    load_drug_token_bundle,
    load_trainable_state,
    predict_scores,
    validate_checkpoint,
)
from external_benchmark_fusion_methods import load_row_indices, row_index_metadata
from external_benchmark_local_protein import ProteinSegmentLookup, load_segment_bundle
from external_benchmark_sepsis import (
    SCENARIOS,
    MatrixSplit,
    atomic_json,
    binary_metrics,
    log,
)
from external_benchmark_atom_segment_multiscale_v7 import (
    MultiScaleAtomSegmentInteractionNet,
    architecture_version as multiscale_v7_architecture_version,
    load_segment_bundle as load_multiscale_v7_segment_bundle,
)
from external_benchmark_atom_segment_calibrated_v8 import (
    CalibratedMultiScaleInteractionNet,
    architecture_version as calibrated_v8_architecture_version,
)
from external_benchmark_atom_segment_ban_v22 import (
    MultiGlimpseBilinearSetInteractionNet,
    architecture_version as bilinear_set_v22_architecture_version,
)
from external_benchmark_atom_segment_energy_v23 import (
    EnergyNormalizedBilinearSetInteractionNet,
    architecture_version as energy_v23_architecture_version,
)
from external_benchmark_atom_segment_energy_v23_no_zero_up import (
    NonZeroUpEnergyNormalizedInteractionNet,
    architecture_version as energy_v23_no_zero_up_architecture_version,
)
from external_benchmark_atom_segment_head_dropout_v28 import (
    HeadDropoutEnergyNormalizedInteractionNet,
    architecture_version as head_dropout_v28_architecture_version,
)
from external_benchmark_atom_segment_mutual_v30 import (
    BidirectionalMutualRefinementInteractionNet,
    architecture_version as mutual_v30_architecture_version,
)
from external_benchmark_atom_segment_weightnorm_v31 import (
    WeightNormalizedBilinearSetInteractionNet,
    architecture_version as weightnorm_v31_architecture_version,
)
from external_benchmark_atom_segment_full_bilinear_v32 import (
    FullBilinearRoutingInteractionNet,
    architecture_version as full_bilinear_v32_architecture_version,
)
from external_benchmark_atom_segment_reciprocal_v33 import (
    ReciprocalBalancedRoutingInteractionNet,
    architecture_version as reciprocal_v33_architecture_version,
)
from external_benchmark_atom_segment_qk_norm_v26 import (
    QKNormalizedRoutingInteractionNet,
    architecture_version as qk_norm_v26_architecture_version,
)
from external_benchmark_atom_segment_mutual_atom_v34 import (
    AtomFromSegmentMutualInteractionNet,
    architecture_version as mutual_atom_v34_architecture_version,
)
from external_benchmark_atom_segment_mutual_segment_v35 import (
    SegmentFromAtomMutualInteractionNet,
    architecture_version as mutual_segment_v35_architecture_version,
)
from external_benchmark_atom_segment_mutual_qk_norm_v36 import (
    BidirectionalMutualQKNormalizedInteractionNet,
    architecture_version as mutual_qk_norm_v36_architecture_version,
)
from external_benchmark_grouped_fusion_v18g import METHOD as GROUPED_V18G_METHOD


def sha256_csv(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / "completed.json"
    if completed_path.exists():
        raise FileExistsError("Test evaluation already completed")

    train_indices = load_row_indices(args.train_indices)
    if train_indices is None:
        raise ValueError("Explicit full training indices are required")
    train_selection = row_index_metadata(args.train_indices, train_indices)
    source = MatrixSplit("{}_train".format(args.scenario), row_indices=train_indices)
    validation = MatrixSplit("{}_valid".format(args.scenario))
    multiscale_variant = args.model_variant in ("multiscale_v7", "calibrated_v8")
    segment_bundle_loader = (
        load_multiscale_v7_segment_bundle if multiscale_variant else load_segment_bundle
    )
    _, _, segment_metadata = segment_bundle_loader(
        args.protein_segments_dir,
        len(source),
        len(validation),
        train_selection,
        args.scenario,
    )
    _, _, token_metadata, token_weights = load_drug_token_bundle(
        args.drug_tokens_dir,
        len(source),
        len(validation),
        train_selection,
        args.scenario,
    )

    global_method = (
        GROUPED_V18G_METHOD
        if args.model_variant == "grouped_atom_v19"
        else "bilinear_control"
    )
    base_checkpoint, base_metadata = load_base_checkpoint(
        args.base_checkpoint, args.seed, train_selection, global_method
    )
    candidate_path = Path(args.candidate_checkpoint).resolve()
    candidate_checkpoint = torch.load(
        candidate_path, map_location="cpu", weights_only=False
    )
    validation_args = argparse.Namespace(
        interaction_dim=args.interaction_dim,
        injection_mode=args.injection_mode,
        seed=args.seed,
        epsilon=args.trained_epsilon,
        residual_energy_weight=args.residual_energy_weight,
        ema_decay=args.ema_decay,
        head_learning_rate=args.head_learning_rate,
        global_method=global_method,
    )
    variant_architecture_version = {
        "multiscale_v7": multiscale_v7_architecture_version,
        "calibrated_v8": calibrated_v8_architecture_version,
        "bilinear_set_v22": bilinear_set_v22_architecture_version,
        "energy_v23": energy_v23_architecture_version,
        "energy_v23_no_zero_up": energy_v23_no_zero_up_architecture_version,
        "head_dropout_v28": head_dropout_v28_architecture_version,
        "mutual_v30": mutual_v30_architecture_version,
        "weightnorm_v31": weightnorm_v31_architecture_version,
        "full_bilinear_v32": full_bilinear_v32_architecture_version,
        "reciprocal_v33": reciprocal_v33_architecture_version,
        "qk_norm_v26": qk_norm_v26_architecture_version,
        "mutual_atom_v34": mutual_atom_v34_architecture_version,
        "mutual_segment_v35": mutual_segment_v35_architecture_version,
        "mutual_qk_norm_v36": mutual_qk_norm_v36_architecture_version,
    }.get(args.model_variant)
    if variant_architecture_version is not None:
        original_architecture_version = atom_segment_base.architecture_version
        atom_segment_base.architecture_version = variant_architecture_version
        try:
            validate_checkpoint(
                candidate_checkpoint,
                validation_args,
                base_metadata,
                segment_metadata,
                token_metadata,
            )
        finally:
            atom_segment_base.architecture_version = original_architecture_version
    else:
        validate_checkpoint(
            candidate_checkpoint,
            validation_args,
            base_metadata,
            segment_metadata,
            token_metadata,
        )

    test = MatrixSplit(args.test_split, preload=args.preload_test)
    test_csv = Path(args.test_csv).resolve()
    test_segment_dir = Path(args.test_protein_segments_dir).resolve()
    with (test_segment_dir / "metadata.json").open(encoding="utf-8") as handle:
        test_segment_metadata = json.load(handle)
    required_test_segment = {
        "scenario": args.test_segment_scenario or args.scenario,
        "test_split": args.test_split,
        "test_rows": len(test),
        "segment_count": 64 if multiscale_variant else 16,
        "protein_dim": 1280,
        "test_labels_used": False,
        "test_csv_sha256": sha256_csv(test_csv),
    }
    for key, expected in required_test_segment.items():
        if test_segment_metadata.get(key) != expected:
            raise ValueError(
                "Test segment metadata {}={} expected {}".format(
                    key, test_segment_metadata.get(key), expected
                )
            )
    test_segments = ProteinSegmentLookup(test_segment_dir, "test", len(test))

    log("Reading explicitly unlocked test Ligand column only")
    test_ligands, csv_rows = read_ligands(test_csv, args.csv_chunk_size)
    if csv_rows != len(test) or len(test_ligands) != len(test):
        raise ValueError("Test CSV and matrix row counts differ")
    unique_smiles, inverse = np.unique(test_ligands, return_inverse=True)

    bermol_model = Path(args.bermol_model).resolve()
    stoi, special_ids, artifact_weights, layer_norm_eps, environment = (
        load_bermol_artifacts(bermol_model, "cpu")
    )
    for key in ("embedding_weight", "layer_norm_weight", "layer_norm_bias"):
        if not np.array_equal(artifact_weights[key], token_weights[key]):
            raise ValueError("BerMol test tokenizer weights differ from training cache")
    if float(token_weights["layer_norm_eps"]) != float(layer_norm_eps):
        raise ValueError("BerMol LayerNorm epsilon differs from training cache")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dtiam-test-drug-", dir="/tmp") as value:
        temporary = Path(value)
        token_stats = build_token_arrays(
            unique_smiles,
            temporary,
            stoi,
            special_ids,
            args.max_tokens,
            args.workers,
            args.tokenize_chunk_size,
            args.progress_every,
            "unk",
            "process",
        )
        np.save(
            temporary / "source_row_codes.npy",
            inverse.astype(np.int32, copy=False),
        )
        test_tokens = DrugTokenLookup(temporary, "source", len(test))

        if args.model_variant == "grouped_atom_v19":
            model = AtomSegmentInteractionNet(
                token_weights,
                token_metadata["padding_token_id"],
                args.interaction_dim,
                args.trained_epsilon,
                args.injection_mode,
                global_method=GROUPED_V18G_METHOD,
            ).to(device)
        else:
            model_class = {
                "standard": AtomSegmentInteractionNet,
                "multiscale_v7": MultiScaleAtomSegmentInteractionNet,
                "calibrated_v8": CalibratedMultiScaleInteractionNet,
                "bilinear_set_v22": MultiGlimpseBilinearSetInteractionNet,
                "energy_v23": EnergyNormalizedBilinearSetInteractionNet,
                "energy_v23_no_zero_up": NonZeroUpEnergyNormalizedInteractionNet,
                "head_dropout_v28": HeadDropoutEnergyNormalizedInteractionNet,
                "mutual_v30": BidirectionalMutualRefinementInteractionNet,
                "weightnorm_v31": WeightNormalizedBilinearSetInteractionNet,
                "full_bilinear_v32": FullBilinearRoutingInteractionNet,
                "reciprocal_v33": ReciprocalBalancedRoutingInteractionNet,
                "qk_norm_v26": QKNormalizedRoutingInteractionNet,
                "mutual_atom_v34": AtomFromSegmentMutualInteractionNet,
                "mutual_segment_v35": SegmentFromAtomMutualInteractionNet,
                "mutual_qk_norm_v36": BidirectionalMutualQKNormalizedInteractionNet,
            }[args.model_variant]
            model = model_class(
                token_weights,
                token_metadata["padding_token_id"],
                args.interaction_dim,
                args.trained_epsilon,
                args.injection_mode,
            ).to(device)
        model.load_frozen_global_state(base_checkpoint["model_state_dict"])
        load_trainable_state(model, candidate_checkpoint["trainable_state_dict"])
        if args.injection_mode == "late_fusion":
            candidate_scores = predict_scores(
                model,
                test,
                test_tokens,
                test_segments,
                device,
                args.batch_size,
                args.amp,
            )
            score_grid = None
        else:
            score_grid = predict_epsilon_grid(
                model,
                test,
                test_tokens,
                test_segments,
                [0.0, args.inference_epsilon],
                device,
                args.batch_size,
                args.amp,
            )

    y_test = np.asarray(test.y[: len(test)], dtype=np.uint8)
    if args.injection_mode == "late_fusion":
        if args.base_test_scores is None:
            raise ValueError("late_fusion requires --base-test-scores")
        zero_scores = np.asarray(
            np.load(args.base_test_scores, allow_pickle=False), dtype=np.float32
        )
    else:
        zero_scores = score_grid["{:.6f}".format(0.0)]
        candidate_scores = score_grid["{:.6f}".format(args.inference_epsilon)]
    base_metrics = binary_metrics(y_test, zero_scores)
    saved_base_metrics = None
    max_base_score_difference = None
    if args.base_test_scores is not None:
        saved_base_scores = np.asarray(
            np.load(args.base_test_scores, allow_pickle=False), dtype=np.float32
        )
        if saved_base_scores.shape != (len(test),):
            raise ValueError("Saved baseline test scores do not align")
        max_base_score_difference = float(
            np.max(np.abs(zero_scores - saved_base_scores))
        )
        if max_base_score_difference > args.baseline_score_tolerance:
            raise AssertionError("Recomputed test baseline does not match saved scores")
        saved_base_metrics = binary_metrics(y_test, saved_base_scores)
    candidate_metrics = binary_metrics(y_test, candidate_scores)
    delta = {
        key: float(candidate_metrics[key] - base_metrics[key])
        for key in ("aupr", "auroc")
    }
    np.save(output_dir / "baseline_scores_recomputed.npy", zero_scores)
    epsilon_token = ("{:.6f}".format(args.inference_epsilon)).replace(".", "").rstrip("0")
    candidate_score_name = (
        "candidate_scores_late_fusion.npy"
        if args.injection_mode == "late_fusion"
        else "candidate_scores_eps{}.npy".format(epsilon_token)
    )
    np.save(output_dir / candidate_score_name, candidate_scores)
    payload = {
        "evaluation": "atom_segment_explicitly_unlocked_test_v1",
        "scenario": args.scenario,
        "test_split": args.test_split,
        "test_csv": str(test_csv),
        "test_csv_sha256": sha256_csv(test_csv),
        "rows": len(test),
        "seed": args.seed,
        "trained_epsilon": args.trained_epsilon,
        "inference_epsilon": args.inference_epsilon,
        "injection_mode": args.injection_mode,
        "model_variant": args.model_variant,
        "candidate_score_file": candidate_score_name,
        "residual_energy_weight": args.residual_energy_weight,
        "ema_decay": args.ema_decay,
        "candidate_validation_weights_source": candidate_checkpoint.get(
            "validation_weights_source", "online"
        ),
        "base_checkpoint": base_metadata,
        "candidate_checkpoint": {
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
            "best_epoch": int(candidate_checkpoint["best_epoch"]),
        },
        "baseline_metrics_same_batch": base_metrics,
        "baseline_metrics_saved_scores": saved_base_metrics,
        "candidate_metrics": candidate_metrics,
        "delta_vs_bn_ln_same_batch": delta,
        "baseline_reproduction_max_abs_score_difference": max_base_score_difference,
        "test_drug_tokenization": token_stats,
        "bermol_environment": environment,
        "test_labels_used_for_evaluation": True,
        "test_data_read": True,
        "hyperparameters_changed_after_test": False,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(completed_path, payload)
    log("Explicit test evaluation complete: {}".format(json.dumps(payload)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=tuple(SCENARIOS), default="protein_cold"
    )
    parser.add_argument("--test-split", default="protein_cold_test")
    parser.add_argument(
        "--test-segment-scenario",
        choices=tuple(SCENARIOS),
        help="Scenario recorded by an externally built test-segment cache.",
    )
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--test-protein-segments-dir", required=True)
    parser.add_argument("--protein-segments-dir", required=True)
    parser.add_argument("--drug-tokens-dir", required=True)
    parser.add_argument("--train-indices", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-test-scores")
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--bermol-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trained-epsilon", type=float, default=0.025)
    parser.add_argument("--inference-epsilon", type=float, default=0.0125)
    parser.add_argument("--residual-energy-weight", type=float, default=0.005)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--head-learning-rate", type=float, default=2e-5)
    parser.add_argument("--interaction-dim", type=int, default=64)
    parser.add_argument(
        "--model-variant",
        choices=[
            "standard", "multiscale_v7", "calibrated_v8", "grouped_atom_v19",
            "bilinear_set_v22",
            "energy_v23",
            "energy_v23_no_zero_up",
            "head_dropout_v28",
            "mutual_v30",
            "weightnorm_v31",
            "full_bilinear_v32",
            "reciprocal_v33",
            "qk_norm_v26",
            "mutual_atom_v34",
            "mutual_segment_v35",
            "mutual_qk_norm_v36",
        ],
        default="standard",
    )
    parser.add_argument(
        "--injection-mode",
        choices=[
            "feature", "logit", "positive_logit", "positive_logit_ste",
            "logit_positive_eval", "late_fusion",
        ],
        default="feature",
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tokenize-chunk-size", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument("--csv-chunk-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--preload-test", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--baseline-score-tolerance", type=float, default=0.005)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
