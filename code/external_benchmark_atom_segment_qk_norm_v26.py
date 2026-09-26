#!/usr/bin/env python3
"""V26 norm-decoupled routing for the V23 atom/segment interaction.

Only the attention routing uses L2-normalized query/key vectors.  Pair
evidence retains the original projected magnitudes.  Attention remains a
latent statistical interaction and is not a physical contact prediction.
"""

import math
from typing import Dict, Tuple

import numpy as np
import torch
from torch.nn import functional as F

import external_benchmark_atom_segment as atom
from external_benchmark_atom_segment_energy_v23 import (
    EnergyNormalizedBilinearSetInteractionNet,
)


METHOD = "atom_segment_qk_normalized_routing_h4_v26"


def architecture_version(
    interaction_dim: int,
    injection_mode: str = "feature",
    global_method: str = "bilinear_control",
) -> str:
    if injection_mode != "feature":
        raise ValueError("V26 supports feature injection only")
    if global_method != "bilinear_control":
        raise ValueError("V26 requires the frozen BN_LN bilinear-control base")
    return (
        "bn_ln+frozen_bermol_atom_esm_segment_multiglimpse_bilinear_set_"
        "energy_normalized_qk_routing_h4_d{}_v26".format(interaction_dim)
    )


ARCHITECTURE_VERSION = architecture_version(64)


class QKNormalizedRoutingInteractionNet(
    EnergyNormalizedBilinearSetInteractionNet
):
    """V23 with norm-independent atom/segment attention routing."""

    def __init__(
        self,
        token_weights: Dict[str, np.ndarray],
        padding_token_id: int,
        interaction_dim: int = 64,
        epsilon: float = 0.025,
        injection_mode: str = "feature",
        global_method: str = "bilinear_control",
    ) -> None:
        super().__init__(
            token_weights,
            padding_token_id,
            interaction_dim,
            epsilon,
            injection_mode,
            global_method,
        )
        self.architecture_version = architecture_version(
            interaction_dim, injection_mode, global_method
        )

    def local_delta(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        segments: torch.Tensor,
        segment_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, token_count = token_ids.shape

        flat_mask = token_mask.reshape(-1)
        valid_token_ids = token_ids.reshape(-1)[flat_mask]
        valid_atoms = self.atom_projection(
            self.token_layer_norm(self.token_embedding(valid_token_ids))
        )
        flat_atoms = valid_atoms.new_zeros(
            (token_ids.numel(), self.interaction_dim)
        )
        flat_atoms[flat_mask] = valid_atoms
        atoms = flat_atoms.reshape(batch_size, token_count, self.interaction_dim)

        atom_mask_f32 = token_mask.unsqueeze(-1).float()
        atom_count = atom_mask_f32.sum(dim=1, keepdim=True).clamp_min(1.0)
        atoms_f32 = atoms.float()
        atom_mean = (atoms_f32 * atom_mask_f32).sum(
            dim=1, keepdim=True
        ) / atom_count
        atoms = ((atoms_f32 - atom_mean) * atom_mask_f32).to(atoms.dtype)

        valid_segments = segment_weights > 0.0
        with torch.autocast(device_type=segments.device.type, enabled=False):
            segment_weights_f32 = segment_weights.float()
            normalized_weights = segment_weights_f32 / segment_weights_f32.sum(
                dim=1, keepdim=True
            ).clamp_min(torch.finfo(torch.float32).tiny)
            segments_f32 = segments.float()
            pooled_segment = torch.einsum(
                "bs,bsf->bf", normalized_weights, segments_f32
            )
            centered_segments = segments_f32 - pooled_segment.unsqueeze(1)
        local_segments = self.segment_projection(centered_segments)
        local_segments = local_segments * valid_segments.unsqueeze(-1).to(
            local_segments.dtype
        )

        atoms_h = atoms.reshape(
            batch_size, token_count, self.interaction_heads, self.head_dim
        )
        segments_h = local_segments.reshape(
            batch_size,
            local_segments.shape[1],
            self.interaction_heads,
            self.head_dim,
        )

        # Decouple routing from projected feature norms.  The diagonal
        # bilinear form remains trainable and controls direction/sign per
        # head; its arbitrary global magnitude can no longer sharpen softmax.
        with torch.autocast(device_type=segments.device.type, enabled=False):
            routed_atoms = atoms_h.float() * self.bilinear_diagonal.float()[
                None, None, :, :
            ]
            queries = F.normalize(routed_atoms, p=2.0, dim=-1, eps=1e-6)
            keys = F.normalize(segments_h.float(), p=2.0, dim=-1, eps=1e-6)
            scores = torch.einsum("bthd,bshd->bhts", queries, keys)
            # A fixed dimension-derived scale keeps the routing-logit order
            # comparable to scaled dot product without adding a tuned or
            # learnable temperature.
            scores = scores * math.sqrt(float(self.head_dim))

        pair_mask = token_mask[:, None, :, None] & valid_segments[:, None, None, :]
        flat_probabilities = self._masked_softmax(
            scores.flatten(2), pair_mask.expand_as(scores).flatten(2), dim=2
        )
        probabilities = flat_probabilities.reshape_as(scores)

        # Values deliberately remain unnormalized: feature magnitude can
        # carry evidence but cannot independently change the routing entropy.
        pair_evidence = torch.einsum(
            "bhts,bthd,bshd->bhd",
            probabilities.to(atoms_h.dtype),
            atoms_h,
            segments_h,
        ).reshape(batch_size, self.interaction_dim)
        delta = self.interaction_mlp(pair_evidence)

        with torch.no_grad():
            atom_marginal = probabilities.float().sum(dim=3)
            segment_marginal = probabilities.float().sum(dim=2)
            tiny = torch.finfo(torch.float32).tiny
            atom_entropy = -(
                atom_marginal.clamp_min(tiny).log() * atom_marginal
            ).sum(dim=2).mean()
            segment_entropy = -(
                segment_marginal.clamp_min(tiny).log() * segment_marginal
            ).sum(dim=2).mean()
        return delta, {
            "atom_attention_entropy": atom_entropy,
            "segment_attention_entropy": segment_entropy,
        }


def install() -> None:
    atom.AtomSegmentInteractionNet = QKNormalizedRoutingInteractionNet
    atom.architecture_version = architecture_version


def main() -> None:
    install()
    atom.train(atom.build_parser().parse_args())


if __name__ == "__main__":
    main()
