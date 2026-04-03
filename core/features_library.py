"""
Mathematical feature extraction library for attention matrices.

This module provides a collection of pure mathematical functions for
computing metrics on attention matrices, query/key tensors, and derived
structures. All functions follow a consistent interface and are registered
in FEATURE_REGISTRY for dynamic invocation.

Key Design Principles:
  - Each feature function is pure and side-effect free.
  - Functions return scalar values (float or np.nan for failures).
  - All computations are numerically stable.
  - Device handling is explicit (SVD on CPU for Apple Silicon compatibility).
"""

from typing import Callable, Dict, Optional
import numpy as np
import torch
import math


# ==============================================================================
# Core Feature Functions
# ==============================================================================

def compute_diagonal_mass(ctx: "HeadContext", band_width: int = 5) -> float:
    """
    Calculate the fraction of attention mass concentrated in a diagonal band.
    
    For a causal/localized attention pattern, mass should concentrate near the
    main diagonal. This metric measures what fraction of the total attention
    mass falls within a band of specified width around the main diagonal.
    
    Args:
        ctx: HeadContext instance containing attention_map.
        band_width: Integer width of the diagonal band (default 5).
    
    Returns:
        float: Fraction of attention mass in the diagonal band (0.0 to 1.0).
              Returns np.nan if computation fails.
    
    Mathematical Definition:
        diagonal_mass = sum(A[i, j] for all |i - j| <= band_width / 2) / sum(A)
        where A is the attention map (post-softmax).
    """
    try:
        A = ctx.attention_map  # shape: (seq_len, seq_len)
        seq_len = A.shape[0]
        half_width = band_width // 2
        
        # Create a mask for the diagonal band
        row_idx = torch.arange(seq_len, device=A.device, dtype=torch.float32)
        col_idx = torch.arange(seq_len, device=A.device, dtype=torch.float32)
        row_idx = row_idx.unsqueeze(1)  # (seq_len, 1)
        col_idx = col_idx.unsqueeze(0)  # (1, seq_len)
        
        band_mask = torch.abs(row_idx - col_idx) <= half_width
        
        # Compute mass within band
        mass_in_band = (A * band_mask.float()).sum()
        total_mass = A.sum()
        
        if total_mass > 0:
            return float((mass_in_band / total_mass).item())
        else:
            return np.nan
    except Exception as e:
        print(f"Error in compute_diagonal_mass: {e}")
        return np.nan


def compute_q_sim_consecutive(ctx: "HeadContext") -> float:
    """
    Compute expected cosine similarity between consecutive query vectors.
    
    This metric measures temporal continuity in the query space. High values
    indicate smooth, continuous query evolution over the sequence, while
    low values indicate abrupt changes.
    
    Args:
        ctx: HeadContext instance containing Q (query matrix).
    
    Returns:
        float: Mean cosine similarity between adjacent queries.
               Returns np.nan if computation fails.
    
    Mathematical Definition:
        E[cos(q_t, q_{t+1})] = mean([cos_sim(Q[i], Q[i+1]) for i in 0..seq_len-2])
    """
    try:
        Q = ctx.Q  # shape: (seq_len, head_dim)
        seq_len = Q.shape[0]
        
        if seq_len < 2:
            return np.nan
        
        # Normalize query vectors
        Q_norm = Q / (torch.norm(Q, dim=1, keepdim=True) + 1e-8)
        
        # Compute cosine similarities between consecutive queries
        sims = []
        for i in range(seq_len - 1):
            sim = (Q_norm[i] * Q_norm[i + 1]).sum()
            sims.append(float(sim.item()))
        
        return np.mean(sims)
    except Exception as e:
        print(f"Error in compute_q_sim_consecutive: {e}")
        return np.nan


def compute_effective_rank_Q(ctx: "HeadContext") -> float:
    """
    Compute the effective rank of the query matrix using Shannon entropy.
    
    The effective rank measures the complexity/variability of the query space.
    High effective rank indicates diverse query patterns, while low effective
    rank suggests redundant or aligned queries.
    
    Args:
        ctx: HeadContext instance containing Q (query matrix).
    
    Returns:
        float: Effective rank (Shannon entropy of normalized singular values).
               Returns np.nan if SVD fails.
    
    Mathematical Definition:
        singular_values = SVD(Q)[1]  (normalized)
        p_i = s_i / sum(s)  (probability distribution)
        effective_rank = exp(-sum(p_i * log(p_i)))
    """
    try:
        Q = ctx.Q  # shape: (seq_len, head_dim)
        
        # Move to CPU for SVD (Apple Silicon MPS doesn't support all SVD operations)
        Q_cpu = Q.cpu()
        
        # Compute singular values
        try:
            singular_vals = torch.linalg.svdvals(Q_cpu)
        except Exception:
            # Fallback: use torch.svd if linalg.svdvals is not available
            _, singular_vals, _ = torch.svd(Q_cpu)
        
        singular_vals = singular_vals.float()
        
        # Normalize to create probability distribution
        singular_vals = singular_vals / (singular_vals.sum() + 1e-8)
        
        # Compute Shannon entropy: H = -sum(p * log(p))
        # Handle zeros by adding small epsilon
        probs = singular_vals[singular_vals > 1e-8]
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        
        return float(entropy.item())
    except Exception as e:
        print(f"Error in compute_effective_rank_Q: {e}")
        return np.nan


def compute_attention_entropy(ctx: "HeadContext") -> float:
    """
    Compute Shannon entropy of the attention map.
    
    This measures the dispersion of attention across the sequence. High entropy
    indicates uniform, dispersed attention, while low entropy indicates
    concentrated attention on a few positions.
    
    Args:
        ctx: HeadContext instance containing attention_map.
    
    Returns:
        float: Shannon entropy of the attention distribution.
               Returns np.nan if computation fails.
    
    Mathematical Definition:
        H = -sum(A[i, j] * log(A[i, j])) for all i, j where A[i, j] > 0
        where A is the attention map (post-softmax, already normalized).
    """
    try:
        A = ctx.attention_map  # shape: (seq_len, seq_len)
        
        # Flatten and filter positive values
        flat_attn = A.flatten()
        positive_attn = flat_attn[flat_attn > 1e-8]
        
        if len(positive_attn) == 0:
            return np.nan
        
        # Compute Shannon entropy: H = -sum(p * log(p))
        entropy = -torch.sum(positive_attn * torch.log(positive_attn + 1e-8))
        
        return float(entropy.item())
    except Exception as e:
        print(f"Error in compute_attention_entropy: {e}")
        return np.nan


def compute_diagonal_mass_5(ctx: "HeadContext") -> float:
    """
    Compute diagonal mass within a band of width 5.
    
    Convenience wrapper around compute_diagonal_mass with band_width=5.
    
    Args:
        ctx: HeadContext instance containing attention_map.
    
    Returns:
        float: Fraction of attention mass in diagonal band of width 5.
    """
    return compute_diagonal_mass(ctx, band_width=5)


def compute_query_key_sim_mean(ctx: "HeadContext") -> float:
    """
    Compute mean cosine similarity between query and key vectors.
    
    This metric measures alignment between queries and keys in the learned
    space. High values indicate well-aligned query-key distributions.
    
    Args:
        ctx: HeadContext instance containing Q and K.
    
    Returns:
        float: Mean cosine similarity between Q and K vectors over sequence.
               Returns np.nan if computation fails.
    """
    try:
        Q = ctx.Q  # shape: (seq_len, head_dim)
        K = ctx.K  # shape: (seq_len, head_dim)
        
        if Q.shape[0] != K.shape[0]:
            return np.nan
        
        # Normalize vectors
        Q_norm = Q / (torch.norm(Q, dim=1, keepdim=True) + 1e-8)
        K_norm = K / (torch.norm(K, dim=1, keepdim=True) + 1e-8)
        
        # Compute element-wise cosine similarities
        sims = (Q_norm * K_norm).sum(dim=1)
        return float(sims.mean().item())
    except Exception as e:
        print(f"Error in compute_query_key_sim_mean: {e}")
        return np.nan


def compute_max_attention_weight(ctx: "HeadContext") -> float:
    """
    Compute the maximum single attention weight.
    
    This measures the concentration of attention at a single position.
    High values (closer to 1.0) indicate peaky attention, while low values
    indicate more diffuse attention patterns.
    
    Args:
        ctx: HeadContext instance containing attention_map.
    
    Returns:
        float: Maximum value in the attention map.
    """
    try:
        A = ctx.attention_map
        return float(A.max().item())
    except Exception as e:
        print(f"Error in compute_max_attention_weight: {e}")
        return np.nan


def compute_attention_variance_per_query(ctx: "HeadContext") -> float:
    """
    Compute mean variance of attention weights across queries.
    
    For each query position, compute the variance of its attention distribution
    across all key positions. High variance means selective attention, while
    low variance means uniform attention patterns.
    
    Args:
        ctx: HeadContext instance containing attention_map.
    
    Returns:
        float: Mean variance across all query positions.
               Returns np.nan if computation fails.
    """
    try:
        A = ctx.attention_map  # shape: (seq_len, seq_len)
        
        # Compute variance for each query (row)
        variances = torch.var(A, dim=1)
        return float(variances.mean().item())
    except Exception as e:
        print(f"Error in compute_attention_variance_per_query: {e}")
        return np.nan


def compute_rank_attention_matrix(ctx: "HeadContext") -> float:
    """
    Compute the effective rank of the attention matrix.
    
    Uses singular value decomposition to estimate the intrinsic dimensionality
    of the attention map. Lower rank indicates redundancy/structure.
    
    Args:
        ctx: HeadContext instance containing attention_map.
    
    Returns:
        float: Effective rank (number of significant singular values).
               Returns np.nan if computation fails.
    """
    try:
        A = ctx.attention_map.cpu()
        
        # Compute singular values
        try:
            singular_vals = torch.linalg.svdvals(A)
        except Exception:
            _, singular_vals, _ = torch.svd(A)
        
        singular_vals = singular_vals.float()
        
        # Count how many singular values are significant (> 1% of max)
        threshold = 0.01 * singular_vals.max()
        effective_rank = (singular_vals > threshold).sum().float()
        
        return float(effective_rank.item())
    except Exception as e:
        print(f"Error in compute_rank_attention_matrix: {e}")
        return np.nan


# ==============================================================================
# Feature Registry
# ==============================================================================

FEATURE_REGISTRY: Dict[str, Callable] = {
    "diagonal_mass_5": compute_diagonal_mass_5,
    "q_sim_consecutive": compute_q_sim_consecutive,
    "effective_rank_Q": compute_effective_rank_Q,
    "attention_entropy": compute_attention_entropy,
    "query_key_sim_mean": compute_query_key_sim_mean,
    "max_attention_weight": compute_max_attention_weight,
    "attention_variance_per_query": compute_attention_variance_per_query,
    "rank_attention_matrix": compute_rank_attention_matrix,
}


def get_all_features(ctx: "HeadContext") -> Dict[str, float]:
    """
    Compute all registered features for a given HeadContext.
    
    Iterates through FEATURE_REGISTRY and computes each feature, gracefully
    handling any failures by returning np.nan for failed features.
    
    Args:
        ctx: HeadContext instance.
    
    Returns:
        Dict[str, float]: Dictionary mapping feature names to their computed values.
                         Failed features are represented as np.nan.
    """
    results = {}
    for feature_name, feature_func in FEATURE_REGISTRY.items():
        try:
            results[feature_name] = feature_func(ctx)
        except Exception as e:
            print(f"Warning: Feature '{feature_name}' failed: {e}")
            results[feature_name] = np.nan
    return results
