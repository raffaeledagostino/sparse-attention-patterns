"""
Context dataclass for encapsulating attention head state.

This module defines the HeadContext dataclass that captures the complete
computational state of a single attention head during analysis. It includes
model metadata, layer/head indices, dimensions, and the core tensors (Q, K,
attention map), along with a cache dictionary for feature computation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch

@dataclass
class HeadContext:
    model_name:     str
    layer_idx:      int
    head_idx:       int
    prompt_len:     int
    H_input:        torch.Tensor
    W_q:            torch.Tensor
    W_k:            torch.Tensor
    W_v:            torch.Tensor
    Q:              torch.Tensor
    K:              torch.Tensor
    attention_map:  torch.Tensor
    rope_theta:     float = 100000.0
    rmsnorm_gamma:  Optional[torch.Tensor] = None
    cache:          Dict[str, Any] = field(default_factory=dict)


    def get_head_dim(self) -> int:
        return self.Q.shape[-1]

    def clear_cache(self) -> None:
        self.cache.clear()

    def get_cached_feature(self, feature_name: str) -> Optional[Any]:
        return self.cache.get(feature_name)

    def set_cached_feature(self, feature_name: str, value: Any) -> None:
        self.cache[feature_name] = value