#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class DynamicLoRAParams:
    A: Tensor
    B: Tensor
    scale: Tensor


@dataclass(frozen=True)
class DynamicLoRATargetSpec:
    key: str
    in_features: int
    out_features: int


class DynamicLoRALinear(nn.Linear):
    def __init__(self, source_linear: nn.Linear, target_key: str):
        super().__init__(
            in_features=source_linear.in_features,
            out_features=source_linear.out_features,
            bias=source_linear.bias is not None,
            device=source_linear.weight.device,
            dtype=source_linear.weight.dtype,
        )
        self.weight.data.copy_(source_linear.weight.data)
        if source_linear.bias is not None and self.bias is not None:
            self.bias.data.copy_(source_linear.bias.data)
        self.target_key = target_key
        self._dynamic_lora: DynamicLoRAParams | None = None

    def set_dynamic_lora(self, params: DynamicLoRAParams | None) -> None:
        self._dynamic_lora = params

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = F.linear(inputs, self.weight, self.bias)
        if self._dynamic_lora is None:
            return outputs

        A = self._dynamic_lora.A.to(dtype=inputs.dtype)
        B = self._dynamic_lora.B.to(dtype=inputs.dtype)
        scale = self._dynamic_lora.scale.to(dtype=inputs.dtype).view(-1, 1, 1)

        rank_space = torch.einsum("bsi,bri->bsr", inputs, A)
        delta = torch.einsum("bsr,bor->bso", rank_space, B)
        return outputs + scale * delta


class DynamicLoRABasisHyperNet(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        rank: int,
        basis_count: int,
        target_specs: list[DynamicLoRATargetSpec],
        scale_init: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.basis_count = basis_count
        self.target_specs = target_specs
        self.scale_init = scale_init

        self.norm = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coeff_head = nn.Linear(hidden_dim, len(target_specs) * (2 * basis_count + 1))
        nn.init.zeros_(self.coeff_head.weight)
        nn.init.zeros_(self.coeff_head.bias)

        self.a_bases = nn.ParameterList(
            [nn.Parameter(torch.empty(basis_count, rank, spec.in_features)) for spec in target_specs]
        )
        self.b_bases = nn.ParameterList(
            [nn.Parameter(torch.empty(basis_count, spec.out_features, rank)) for spec in target_specs]
        )

        for basis in self.a_bases:
            nn.init.normal_(basis, std=0.02)
        for basis in self.b_bases:
            nn.init.normal_(basis, std=0.02)

    def forward(self, latent: Tensor) -> dict[str, DynamicLoRAParams]:
        hidden = self.mlp(self.norm(latent))
        coeffs = self.coeff_head(hidden)
        coeffs = coeffs.view(latent.shape[0], len(self.target_specs), 2 * self.basis_count + 1)

        params: dict[str, DynamicLoRAParams] = {}
        for target_idx, spec in enumerate(self.target_specs):
            alpha_A = coeffs[:, target_idx, : self.basis_count]
            alpha_B = coeffs[:, target_idx, self.basis_count : 2 * self.basis_count]
            scale = self.scale_init + coeffs[:, target_idx, -1]

            A = torch.einsum("bm,mri->bri", alpha_A, self.a_bases[target_idx])
            B = torch.einsum("bm,mor->bor", alpha_B, self.b_bases[target_idx])
            params[spec.key] = DynamicLoRAParams(A=A, B=B, scale=scale)

        return params