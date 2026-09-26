#!/usr/bin/env python3
"""Bidirectional mutual refinement followed by Q/K-normalized pair routing.

The first cross-attention stage is identical to the established bidirectional
mutual-refinement architecture. Its zero-initialized scalar gates update atom
and protein-segment features in both directions. Only the final pair-routing
softmax uses L2-normalized queries and keys; the value/evidence path preserves
the refined feature magnitudes.
"""

import math
from typing import Dict, Tuple

import numpy as np
import torch
from torch.nn import functional as F

import external_benchmark_atom_segment as atom
from external_benchmark_atom_segment_mutual_v30 import (
    BidirectionalMutualRefinementInteractionNet,
)


METHOD = "atom_segment_bidirectional_mutual_qk_normalized_routing_h4_v36"


def architecture_version(
    interaction_dim: int,
    injection_mode: str = "feature",
    global_method: str = "bilinear_control",
) -> str:
    if injection_mode != "feature":
        raise ValueError("The combined architecture supports feature injection only")
    if global_method != "bilinear_control":
        raise ValueError("The combined architecture requires the frozen BN_LN base")
    return (
        "bn_ln+frozen_bermol_atom_esm_segment_bidirectional_mutual_refine_"
        "gated_zero_init_then_multiglimpse_bilinear_set_energy_normalized_"
        "qk_routing_h4_d{}_v36".format(interaction_dim)
    )


ARCHITECTURE_VERSION = architecture_version(64)


class BidirectionalMutualQKNormalizedInteractionNet(
    BidirectionalMutualRefinementInteractionNet
):
    """Mutually refine both modalities, then use norm-independent routing."""

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

        # Keep the established mutual-refinement stage unchanged.
        initial_scores = torch.einsum(
            "bthd,bshd,hd->bhts",
            atoms_h,
            segments_h,
            self.bilinear_diagonal,
        ) / math.sqrt(float(self.head_dim))
        pair_mask = token_mask[:, None, :, None] & valid_segments[:, None, None, :]
        expanded_mask = pair_mask.expand_as(initial_scores)
        atom_to_segment = self._masked_softmax(
            initial_scores, expanded_mask, dim=3
        )
        segment_to_atom = self._masked_softmax(
            initial_scores, expanded_mask, dim=2
        )
        atom_context = torch.einsum(
            "bhts,bshd->bthd", atom_to_segment, segments_h
        )
        segment_context = torch.einsum(
            "bhts,bthd->bshd", segment_to_atom, atoms_h
        )
        atom_gate = torch.tanh(self.atom_mutual_gate).to(atoms_h.dtype)
        segment_gate = torch.tanh(self.segment_mutual_gate).to(segments_h.dtype)
        refined_atoms = atoms_h + atom_gate * atom_context
        refined_segments = segments_h + segment_gate * segment_context
        refined_atoms = refined_atoms * token_mask[:, :, None, None].to(
            refined_atoms.dtype
        )
        refined_segments = refined_segments * valid_segments[:, :, None, None].to(
            refined_segments.dtype
        )

        # Normalize only final-routing Q/K. Values retain their magnitudes.
        with torch.autocast(device_type=segments.device.type, enabled=False):
            routed_atoms = refined_atoms.float() * self.bilinear_diagonal.float()[
                None, None, :, :
            ]
            queries = F.normalize(routed_atoms, p=2.0, dim=-1, eps=1e-6)
            keys = F.normalize(
                refined_segments.float(), p=2.0, dim=-1, eps=1e-6
            )
            scores = torch.einsum("bthd,bshd->bhts", queries, keys)
            scores = scores * math.sqrt(float(self.head_dim))

        flat_probabilities = self._masked_softmax(
            scores.flatten(2), expanded_mask.flatten(2), dim=2
        )
        probabilities = flat_probabilities.reshape_as(scores)
        pair_evidence = torch.einsum(
            "bhts,bthd,bshd->bhd",
            probabilities.to(refined_atoms.dtype),
            refined_atoms,
            refined_segments,
        ).reshape(batch_size, self.interaction_dim)
        delta = self.interaction_mlp(pair_evidence)

        diagnostics = self.interaction_diagnostics(probabilities)
        diagnostics.update(
            {
                "atom_mutual_gate": atom_gate.detach().float(),
                "segment_mutual_gate": segment_gate.detach().float(),
            }
        )
        return delta, diagnostics


def install() -> None:
    atom.AtomSegmentInteractionNet = BidirectionalMutualQKNormalizedInteractionNet
    atom.architecture_version = architecture_version


def main() -> None:
    install()
    atom.train(atom.build_parser().parse_args())


if __name__ == "__main__":
    main()
