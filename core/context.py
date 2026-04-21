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

# core/context.py
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
    rmsnorm_gamma:  Optional[torch.Tensor] = None
    cache:          Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Normalize all tensors to float32 on CPU at construction time.
        This is the single point of dtype normalization for the entire pipeline.
        bfloat16 (Qwen3 native) and float16 are both cast to float32 here.
        """
        _to_f32 = lambda t: t.detach().cpu().float() if t is not None else None

        self.H_input       = _to_f32(self.H_input)
        self.W_q           = _to_f32(self.W_q)
        self.W_k           = _to_f32(self.W_k)
        self.W_v           = _to_f32(self.W_v)
        self.Q             = _to_f32(self.Q)
        self.K             = _to_f32(self.K)
        self.attention_map = _to_f32(self.attention_map)
        self.rmsnorm_gamma = _to_f32(self.rmsnorm_gamma)
    
    def get_head_dim(self) -> int:
        """
        Get the dimension of a single attention head.
        
        Returns:
            int: The feature dimension of this head (typically hidden_dim / num_heads).
        """
        return self.Q.shape[-1]
    
    def clear_cache(self) -> None:
        """Clear the feature cache to free memory."""
        self.cache.clear()
    
    def get_cached_feature(self, feature_name: str) -> Optional[Any]:
        """
        Retrieve a cached feature value.
        
        Args:
            feature_name (str): Name of the feature to retrieve.
            
        Returns:
            Optional[Any]: The cached feature value, or None if not in cache.
        """
        return self.cache.get(feature_name)
    
    def set_cached_feature(self, feature_name: str, value: Any) -> None:
        """
        Store a computed feature in the cache.
        
        Args:
            feature_name (str): Name of the feature.
            value (Any): The computed feature value.
        """
        self.cache[feature_name] = value
