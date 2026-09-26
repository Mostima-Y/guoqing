#!/usr/bin/env python3
"""Evaluate one locked V23/V36 checkpoint on the existing validation split."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import evaluate_atom_segment_test as shared


MODEL_CLASSES = {
    "energy_v23": shared.EnergyNormalizedBilinearSetInteractionNet,
    "mutual_qk_norm_v36": shared.BidirectionalMutualQKNormalizedInteractionNet,
}
ARCHITECTURE_VERSIONS = {
    "energy_v23": shared.energy_v23_architecture_version,
    "mutual_qk_norm_v36": shared.mutual_qk_norm_v36_architecture_version,
}


def epsilon_token(epsilon: float) -> str:
    return ("{:.6f}".format(epsilon)).replace(".", "").rstrip("0")


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    train_indices = shared.load_row_indices(args.train_indices)
    if train_indices is None:
        raise ValueError("Explicit full training indices are required")
    train_selection = shared.row_index_metadata(args.train_indices, train_indices)
    source = shared.MatrixSplit(
        "{}_train".format(args.scenario), row_indices=train_indices
    )
    validation_split = "{}_valid".format(args.scenario)
    validation = shared.MatrixSplit(
        validation_split, preload=args.preload_validation
    )
    _, validation_segments, segment_metadata = shared.load_segment_bundle(
        args.protein_segments_dir,
        len(source),
        len(validation),
        train_selection,
        args.scenario,
    )
    _, validation_tokens, token_metadata, token_weights = shared.load_drug_token_bundle(
        args.drug_tokens_dir,
        len(source),
        len(validation),
        train_selection,
        args.scenario,
    )

    global_method = "bilinear_control"
    base_checkpoint, base_metadata = shared.load_base_checkpoint(
        args.base_checkpoint, args.seed, train_selection, global_method
    )
    candidate_path = Path(args.candidate_checkpoint).resolve()
    candidate_checkpoint = torch.load(
        candidate_path, map_location="cpu", weights_only=False
    )
    validation_args = argparse.Namespace(
        interaction_dim=args.interaction_dim,
        injection_mode="feature",
        seed=args.seed,
        epsilon=args.trained_epsilon,
        residual_energy_weight=args.residual_energy_weight,
        ema_decay=args.ema_decay,
        head_learning_rate=args.head_learning_rate,
        global_method=global_method,
    )
    original_architecture_version = shared.atom_segment_base.architecture_version
    shared.atom_segment_base.architecture_version = ARCHITECTURE_VERSIONS[
        args.model_variant
    ]
    try:
        shared.validate_checkpoint(
            candidate_checkpoint,
            validation_args,
            base_metadata,
            segment_metadata,
            token_metadata,
        )
    finally:
        shared.atom_segment_base.architecture_version = original_architecture_version

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)

    model = MODEL_CLASSES[args.model_variant](
        token_weights,
        token_metadata["padding_token_id"],
        args.interaction_dim,
        args.trained_epsilon,
        "feature",
    ).to(device)
    model.load_frozen_global_state(base_checkpoint["model_state_dict"])
    shared.load_trainable_state(model, candidate_checkpoint["trainable_state_dict"])

    inference_epsilons = tuple(dict.fromkeys(args.inference_epsilons))
    if any(epsilon <= 0.0 for epsilon in inference_epsilons):
        raise ValueError("Only positive inference epsilons are allowed")
    started = time.perf_counter()
    score_grid = shared.predict_epsilon_grid(
        model,
        validation,
        validation_tokens,
        validation_segments,
        [0.0, *inference_epsilons],
        device,
        args.batch_size,
        args.amp,
    )

    y_validation = np.asarray(validation.y[: len(validation)], dtype=np.uint8)
    zero_scores = score_grid["{:.6f}".format(0.0)]
    np.save(output_dir / "baseline_scores_recomputed.npy", zero_scores)
    candidate_metrics = {}
    candidate_score_files = {}
    for epsilon in inference_epsilons:
        score_key = "{:.6f}".format(epsilon)
        scores = score_grid[score_key]
        score_name = "candidate_scores_eps{}.npy".format(epsilon_token(epsilon))
        np.save(output_dir / score_name, scores)
        candidate_metrics[score_key] = shared.binary_metrics(y_validation, scores)
        candidate_score_files[score_key] = score_name

    payload = {
        "evaluation": "atom_segment_locked_validation_epsilon_grid_v1",
        "scenario": args.scenario,
        "evaluation_split": validation_split,
        "rows": len(validation),
        "seed": args.seed,
        "trained_epsilon": args.trained_epsilon,
        "inference_epsilons": list(inference_epsilons),
        "model_variant": args.model_variant,
        "candidate_score_files": candidate_score_files,
        "candidate_metrics": candidate_metrics,
        "baseline_metrics_same_batch": shared.binary_metrics(
            y_validation, zero_scores
        ),
        "residual_energy_weight": args.residual_energy_weight,
        "ema_decay": args.ema_decay,
        "base_checkpoint": base_metadata,
        "candidate_checkpoint": {
            "path": str(candidate_path),
            "sha256": shared.file_sha256(candidate_path),
            "best_epoch": int(candidate_checkpoint["best_epoch"]),
        },
        "selection_note": "Existing selected checkpoint; no validation reselection.",
        "test_data_read": False,
        "seconds": time.perf_counter() - started,
    }
    shared.atomic_json(output_dir / "completed.json", payload)
    shared.log("Validation epsilon evaluation complete: {}".format(json.dumps(payload)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("protein_cold",), default="protein_cold")
    parser.add_argument("--protein-segments-dir", required=True)
    parser.add_argument("--drug-tokens-dir", required=True)
    parser.add_argument("--train-indices", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--trained-epsilon", type=float, default=0.025)
    parser.add_argument(
        "--inference-epsilons",
        type=float,
        nargs="+",
        default=(0.025, 0.0125),
    )
    parser.add_argument("--residual-energy-weight", type=float, default=0.005)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--head-learning-rate", type=float, default=2e-5)
    parser.add_argument("--interaction-dim", type=int, default=64)
    parser.add_argument(
        "--model-variant", choices=tuple(MODEL_CLASSES), required=True
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--preload-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
