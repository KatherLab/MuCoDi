# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Adapted from Facebook Research MoCo v3 under CC BY-NC 4.0.
# See NOTICE and LICENSES/CC-BY-NC-4.0.txt.

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather


class KDInfoNCELoss(nn.Module):
    """MoCo-v3 InfoNCE using cached teacher features as fixed keys."""

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.tau = temperature

    def gather_keys(self, k_pos: torch.Tensor):
        """Normalize teacher keys and gather them across distributed workers."""
        k = F.normalize(k_pos, dim=-1)

        if dist.is_available() and dist.is_initialized():
            k_all = torch.cat(list(all_gather(k)), dim=0)
            local_batch_size = k_pos.size(0)
            labels = (
                dist.get_rank() * local_batch_size
                + torch.arange(local_batch_size, device=k_pos.device)
            )
        else:
            k_all = k
            labels = torch.arange(k_pos.size(0), device=k_pos.device)

        return k_all, labels

    def forward_single(
        self,
        q: torch.Tensor,
        k_all: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE for one student view and one teacher."""
        q = F.normalize(q, dim=-1)
        logits = (q @ k_all.T) / self.tau
        return 2 * self.tau * F.cross_entropy(logits, labels)
