#!/usr/bin/env python3
"""V23 energy-normalized multi-glimpse atom/segment residual."""

from typing import Dict

import numpy as np
import torch

import external_benchmark_atom_segment as atom
from external_benchmark_atom_segment_ban_v22 import (
    MultiGlimpseBilinearSetInteractionNet,
)


def architecture_version(
    interaction_dim: int,
    injection_mode: str = "feature",
    global_method: str = "bilinear_control",
) -> str:
    if injection_mode != "feature":
        raise ValueError("V23 supports feature injection only")
    if global_method != "bilinear_control":
        raise ValueError("V23 requires the frozen BN_LN bilinear-control base")
    return (
        "bn_ln+frozen_bermol_atom_esm_segment_multiglimpse_bilinear_set_"
        "energy_normalized_h4_d{}_v23".format(interaction_dim)
    )


ARCHITECTURE_VERSION = architecture_version(64)


class EnergyNormalizedBilinearSetInteractionNet(
    MultiGlimpseBilinearSetInteractionNet
):
    """V22 pair representation with parameter-free residual energy control."""

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

    def bounded_residual(self, delta: torch.Tensor) -> torch.Tensor:
        # The scale equals one at the exact-zero initialization, preserving a
        # nonzero first derivative.  As raw branch energy grows, the injected
        # feature residual is smoothly attenuated rather than saturating at a
        # fixed tanh magnitude.  There is no learned gate or tuned threshold.
        delta_f32 = delta.float()
        inverse_energy = torch.rsqrt(
            1.0 + delta_f32.square().mean(dim=-1, keepdim=True)
        )
        return torch.tanh(delta) * inverse_energy.to(delta.dtype)


def install() -> None:
    atom.AtomSegmentInteractionNet = EnergyNormalizedBilinearSetInteractionNet
    atom.architecture_version = architecture_version


def main() -> None:
    install()
    atom.train(atom.build_parser().parse_args())


if __name__ == "__main__":
    main()
