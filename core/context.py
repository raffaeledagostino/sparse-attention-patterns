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
    """
    Encapsulates the complete state of a single attention head.
    
    This dataclass holds all necessary information for feature extraction
    from a single attention head, including model metadata, tensor references,
    and a cache for computed metrics to avoid redundant calculations.
    
    Attributes:
        model_name (str): Name of the LLM model (e.g., "Qwen/Qwen2.5-0.5B-Instruct").
        layer_idx (int): Index of the transformer layer (0-indexed).
        head_idx (int): Index of the attention head within the layer.
        prompt_len (int): Sequence length of the processed prompt.
        H_input (torch.Tensor): Hidden state input to this layer, shape (seq_len, hidden_dim).
        W_q (torch.Tensor): Query weight projection matrix, shape (hidden_dim, hidden_dim).
        W_k (torch.Tensor): Key weight projection matrix, shape (hidden_dim, hidden_dim).
        Q (torch.Tensor): Query matrix (per-head), shape (seq_len, head_dim).
        K (torch.Tensor): Key matrix (per-head), shape (seq_len, head_dim).
        attention_map (torch.Tensor): Softmax attention weights, shape (seq_len, seq_len).
        cache (Dict[str, Any]): Dictionary for caching computed features to avoid redundant calculations.
    """
    
    model_name: str
    layer_idx: int
    head_idx: int
    prompt_len: int
    H_input: torch.Tensor
    W_q: torch.Tensor
    W_k: torch.Tensor
    Q: torch.Tensor
    K: torch.Tensor
    attention_map: torch.Tensor
    cache: Dict[str, Any] = field(default_factory=dict)
    
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
