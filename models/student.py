from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


def make_projector(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
) -> nn.Module:
    """Create a teacher-specific linear head or MoCo-v3-style MLP."""
    if num_layers <= 1:
        return nn.Linear(in_dim, out_dim, bias=True)

    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim, bias=True),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
    ]

    for _ in range(num_layers - 2):
        layers.extend(
            [
                nn.Linear(hidden_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        )

    # Final projection intentionally has no batch normalization or activation.
    layers.append(nn.Linear(hidden_dim, out_dim, bias=True))
    return nn.Sequential(*layers)


class MultiTeacherStudent(nn.Module):
    """Student backbone with one projection head for each teacher."""

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        teacher_dims: Mapping[str, int],
        hidden: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projectors = nn.ModuleDict(
            {
                name: make_projector(
                    in_dim=feat_dim,
                    hidden_dim=hidden,
                    out_dim=teacher_dim,
                    num_layers=num_layers,
                )
                for name, teacher_dim in teacher_dims.items()
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the projected student representation for every teacher."""
        features = self.backbone(x)
        return {
            name: projector(features)
            for name, projector in self.projectors.items()
        }
