"""
Mathematical feature extraction library for attention matrices.

This module provides a collection of pure mathematical functions for
computing metrics on attention matrices, query/key tensors, weight matrices,
and hidden states. All functions follow a consistent interface and are
registered in FEATURE_REGISTRY for dynamic invocation.

Key Design Principles:
  - Each feature function is pure and side-effect free.
  - Functions return scalar floats (np.nan on failure).
  - SVD computations are always dispatched to CPU for Apple Silicon compatibility.
  - ctx.cache is used to memoize SVD results within a single (layer, head) call.
  - FEATURE_REGISTRY is the single source of truth: add a function here only.
"""

from typing import Callable, Dict, Tuple
import numpy as np
import torch

from core.context import HeadContext


# ==============================================================================
# SVD Infrastructure — single decomposition per matrix, shared across features
# ==============================================================================

SVD_COMPUTE_DTYPE = torch.float32

def _to_svd_tensor(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.detach().cpu().to(dtype=SVD_COMPUTE_DTYPE)

def _economy_svd(matrix: torch.Tensor):
    """
    Economy SVD on CPU float32. Returns (U, S, Vh) with full_matrices=False.
    This is the ONE place where torch.linalg.svd is called.
    full_matrices=False: for (m,n) with m<n returns U(m,m), S(m), Vh(m,n).
    ~3x faster than full SVD for rectangular matrices like W_q (64x896).
    """
    m = _to_svd_tensor(matrix)
    try:
        return torch.linalg.svd(m, full_matrices=False)
    except Exception:
        U, S, V = torch.svd(m, some=True)
        return U, S, V.T

def _get_cached_svd(ctx: "HeadContext", key: str, matrix: torch.Tensor):
    """
    Cache full economy SVD. Key convention: 'svd_Wq', 'svd_Wk', 'svd_Q', etc.
    Compute once, reuse for both rank metrics AND alignment features.
    """
    if key not in ctx.cache:
        ctx.cache[key] = _economy_svd(matrix)
    return ctx.cache[key]

def _rank_metrics_from_S(S: torch.Tensor) -> Dict[str, float]:
    """Compute effective_rank and r95 from precomputed singular values."""
    total = S.sum() + 1e-12
    probs = S / total
    p_nz = probs[probs > 1e-12]
    entropy = -torch.sum(p_nz * torch.log(p_nz))
    cumsum = torch.cumsum(probs, dim=0)
    return {
        "effective_rank": float(torch.exp(entropy).item()),
        "r95": int((cumsum < 0.95).sum().item()) + 1,
    }

def _get_cached_rank(ctx: "HeadContext", svd_key: str, matrix: torch.Tensor) -> Dict[str, float]:
    """
    Rank metrics from cached SVD. svd_key must match _get_cached_svd key.
    rank_key is derived as 'rank_' + svd_key to avoid collision.
    """
    rank_key = "rank_" + svd_key
    if rank_key not in ctx.cache:
        _, S, _ = _get_cached_svd(ctx, svd_key, matrix)
        ctx.cache[rank_key] = _rank_metrics_from_S(S)
    return ctx.cache[rank_key]


# ==============================================================================
# Rank of Weight Matrices (W_q, W_k, W_v)
# ==============================================================================


def compute_effective_rank_Wq(ctx: "HeadContext") -> float:
    try:    return _get_cached_rank(ctx, 'svd_Wq', ctx.W_q)['effective_rank']
    except Exception as e: print(f"Error in compute_effective_rank_Wq: {e}"); return np.nan

def compute_r95_Wq(ctx: "HeadContext") -> float:
    try:    return float(_get_cached_rank(ctx, 'svd_Wq', ctx.W_q)['r95'])
    except Exception as e: print(f"Error in compute_r95_Wq: {e}"); return np.nan

def compute_effective_rank_Wk(ctx: "HeadContext") -> float:
    try:    return _get_cached_rank(ctx, 'svd_Wk', ctx.W_k)['effective_rank']
    except Exception as e: print(f"Error in compute_effective_rank_Wk: {e}"); return np.nan

def compute_r95_Wk(ctx: "HeadContext") -> float:
    try:    return float(_get_cached_rank(ctx, 'svd_Wk', ctx.W_k)['r95'])
    except Exception as e: print(f"Error in compute_r95_Wk: {e}"); return np.nan

def compute_effective_rank_Wv(ctx: "HeadContext") -> float:
    try:    return _get_cached_rank(ctx, 'svd_Wv', ctx.W_v)['effective_rank']
    except Exception as e: print(f"Error in compute_effective_rank_Wv: {e}"); return np.nan

def compute_r95_Wv(ctx: "HeadContext") -> float:
    try:    return float(_get_cached_rank(ctx, 'svd_Wv', ctx.W_v)['r95'])
    except Exception as e: print(f"Error in compute_r95_Wv: {e}"); return np.nan


# ==============================================================================
# Rank of Hidden States H
# ==============================================================================

def compute_effective_rank_H(ctx: "HeadContext") -> float:
    try:    return _get_cached_rank(ctx, 'svd_H', ctx.H_input)['effective_rank']
    except Exception as e: print(f"Error in compute_effective_rank_H: {e}"); return np.nan

def compute_r95_H(ctx: "HeadContext") -> float:
    try:    return float(_get_cached_rank(ctx, 'svd_H', ctx.H_input)['r95'])
    except Exception as e: print(f"Error in compute_r95_H: {e}"); return np.nan


# ==============================================================================
# Rank of Projected Q and K
# ==============================================================================

def compute_effective_rank_Q(ctx: "HeadContext") -> float:
    try:    return _get_cached_rank(ctx, 'svd_Q', ctx.Q)['effective_rank']
    except Exception as e: print(f"Error in compute_effective_rank_Q: {e}"); return np.nan

def compute_r95_Q(ctx: "HeadContext") -> float:
    try:    return float(_get_cached_rank(ctx, 'svd_Q', ctx.Q)['r95'])
    except Exception as e: print(f"Error in compute_r95_Q: {e}"); return np.nan

def compute_effective_rank_K(ctx: "HeadContext") -> float:
    try:    return _get_cached_rank(ctx, 'svd_K', ctx.K)['effective_rank']
    except Exception as e: print(f"Error in compute_effective_rank_K: {e}"); return np.nan

def compute_r95_K(ctx: "HeadContext") -> float:
    try:    return float(_get_cached_rank(ctx, 'svd_K', ctx.K)['r95'])
    except Exception as e: print(f"Error in compute_r95_K: {e}"); return np.nan


# ==============================================================================
# Temporal Similarity (Q and K consecutive similarity)
# ==============================================================================

def compute_q_sim_consecutive(ctx: "HeadContext") -> float:
    """
    Expected cosine similarity between temporally adjacent query vectors.

    Mathematical Definition:
        E[cos(q_t, q_{t+1})] = mean(cos_sim(Q[:-1], Q[1:]))
    """
    try:
        Q = ctx.Q
        if Q.shape[0] < 2:
            return np.nan
        Q_norm = Q / (torch.norm(Q, dim=1, keepdim=True) + 1e-8)
        sims = (Q_norm[:-1] * Q_norm[1:]).sum(dim=1)
        return float(sims.mean().item())
    except Exception as e:
        print(f"Error in compute_q_sim_consecutive: {e}")
        return np.nan


def compute_k_sim_consecutive(ctx: "HeadContext") -> float:
    """
    Expected cosine similarity between temporally adjacent key vectors.

    Mathematical Definition:
        E[cos(k_t, k_{t+1})] = mean(cos_sim(K[:-1], K[1:]))
    """
    try:
        K = ctx.K
        if K.shape[0] < 2:
            return np.nan
        K_norm = K / (torch.norm(K, dim=1, keepdim=True) + 1e-8)
        sims = (K_norm[:-1] * K_norm[1:]).sum(dim=1)
        return float(sims.mean().item())
    except Exception as e:
        print(f"Error in compute_k_sim_consecutive: {e}")
        return np.nan


# ==============================================================================
# SVD Alignment (H vs W_q, H vs W_k)
# ==============================================================================


def build_rope_rotation(delta: int, d_head: int, theta_base: float = 10000.0) -> torch.Tensor:
    """
    Build the RoPE rotation matrix R_{Δθ} ∈ R^{d_head × d_head} for a relative
    position delta.

    The matrix acts on 2D subspaces (pairs of dimensions). delta=0 yields the
    identity matrix.
    """
    half = d_head // 2
    k = torch.arange(half, dtype=torch.float32)
    angles = (theta_base ** (-2.0 * k / d_head)) * delta
    cos_a, sin_a = angles.cos(), angles.sin()

    idx = k.long() * 2
    R = torch.zeros(d_head, d_head)
    R[idx, idx] = cos_a
    R[idx, idx + 1] = -sin_a
    R[idx + 1, idx] = sin_a
    R[idx + 1, idx + 1] = cos_a
    return R

def compute_svd_alignment_H_Wq(ctx: "HeadContext") -> float:
    """Riusa SVD già in cache — nessuna decomposizione aggiuntiva."""
    try:
        _, _, Vh_H  = _get_cached_svd(ctx, 'svd_H',  ctx.H_input)
        _, _, Vh_Wq = _get_cached_svd(ctx, 'svd_Wq', ctx.W_q)
        G = Vh_H[:2].cpu() @ Vh_Wq[:2].cpu().T
        return float(torch.linalg.svdvals(G.float()).mean().item())
    except Exception as e:
        print(f"Error in compute_svd_alignment_H_Wq: {e}"); return np.nan

def compute_svd_alignment_H_Wk(ctx: "HeadContext") -> float:
    try:
        _, _, Vh_H  = _get_cached_svd(ctx, 'svd_H',  ctx.H_input)
        _, _, Vh_Wk = _get_cached_svd(ctx, 'svd_Wk', ctx.W_k)
        G = Vh_H[:2].cpu() @ Vh_Wk[:2].cpu().T
        return float(torch.linalg.svdvals(G.float()).mean().item())
    except Exception as e:
        print(f"Error in compute_svd_alignment_H_Wk: {e}"); return np.nan
    
def _compute_WqRWk_alignment(ctx: "HeadContext", delta: int) -> float:
    try:
        cache_key = f"svd_WqRWk_delta_{delta}"
        if cache_key not in ctx.cache:
            if "WqWk" not in ctx.cache:
                ctx.cache["WqWk"] = _to_svd_tensor(ctx.W_q) @ _to_svd_tensor(ctx.W_k).T
            WqWk = ctx.cache["WqWk"]
            d_head = WqWk.shape[0]
            R = build_rope_rotation(delta, d_head, ctx.rope_theta).to(WqWk.dtype)
            M = WqWk @ R.T
            ctx.cache[cache_key] = _economy_svd(M)

        U, S, Vh = ctx.cache[cache_key]
        V        = Vh.T
        cos_sim  = (U * V).sum(dim=0)
        weights  = S / (S.sum() + 1e-12)
        return float((weights * cos_sim).sum().item())
    except Exception as e:
        print(f"Error in _compute_WqRWk_alignment delta={delta}: {e}")
        return np.nan

def compute_WqRWk_alignment_delta_0(ctx: "HeadContext") -> float:
    """QK alignment, Δ=0 (R=I). Equivalent to the old W_q/W_k alignment."""
    try:
        return _compute_WqRWk_alignment(ctx, delta=0)
    except Exception as e:
        print(f"Error in compute_WqRWk_alignment_delta_0: {e}")
        return np.nan


def compute_WqRWk_alignment_delta_1(ctx: "HeadContext") -> float:
    """QK alignment for tokens 1 step apart (Δ=1)."""
    try:
        return _compute_WqRWk_alignment(ctx, delta=1)
    except Exception as e:
        print(f"Error in compute_WqRWk_alignment_delta_1: {e}")
        return np.nan


def compute_WqRWk_alignment_delta_2(ctx: "HeadContext") -> float:
    """QK alignment for tokens 2 steps apart (Δ=2)."""
    try:
        return _compute_WqRWk_alignment(ctx, delta=2)
    except Exception as e:
        print(f"Error in compute_WqRWk_alignment_delta_2: {e}")
        return np.nan


def compute_WqRWk_alignment_delta_3(ctx: "HeadContext") -> float:
    """QK alignment for tokens 3 steps apart (Δ=3)."""
    try:
        return _compute_WqRWk_alignment(ctx, delta=3)
    except Exception as e:
        print(f"Error in compute_WqRWk_alignment_delta_3: {e}")
        return np.nan


def compute_WqRWk_alignment_delta_4(ctx: "HeadContext") -> float:
    """QK alignment for tokens 4 steps apart (Δ=4)."""
    try:
        return _compute_WqRWk_alignment(ctx, delta=4)
    except Exception as e:
        print(f"Error in compute_WqRWk_alignment_delta_4: {e}")
        return np.nan
    

# ==============================================================================
# RMSNorm Gamma and Channel-Wise Variance and Center of Mass of RoPE Frequencies
# ==============================================================================

def compute_rmsnorm_gamma_norm(ctx: "HeadContext") -> float:
    """
    L2 norm of the RMSNorm gamma (scale) parameter vector.

    Measures the overall magnitude of the learned channel-wise rescaling
    applied to hidden states before the attention projection. Returns np.nan
    if the model does not use QK-Norm or if gamma is not stored in the context.
    """
    try:
        if ctx.rmsnorm_gamma is None:
            return np.nan
        gamma = ctx.rmsnorm_gamma.detach().cpu().float()
        return float(torch.norm(gamma).item())
    except Exception as e:
        print(f"Error in compute_rmsnorm_gamma_norm: {e}")
        return np.nan

def _rope_pair_norms(W: torch.Tensor) -> torch.Tensor:
    """
    Compute RoPE-pair norms for a projection matrix W of shape (d_h, d_model).

    RoPE rotates dimensions in consecutive pairs (2k, 2k+1) with the same
    angular frequency theta_k. The natural unit of analysis is therefore
    the pair norm:
        c_k = sqrt(||W[:, 2k]||^2 + ||W[:, 2k+1]||^2)

    Returns:
        Tensor of shape (d_h // 2,) with one norm per frequency pair.
    """
    m = W.detach().cpu().float()
    
    row_norms_sq = (m ** 2).sum(dim=1)           # (d_h,)
    pair_norms_sq = row_norms_sq.view(-1, 2).sum(dim=1)  # (d_h//2,)
    return pair_norms_sq.sqrt()                           # (d_h//2,)


def _get_cached_rope_pair_norms(ctx: "HeadContext", key: str, W: torch.Tensor) -> torch.Tensor:
    """Cache RoPE pair norms — called up to 4 times per W per head."""
    if key not in ctx.cache:
        ctx.cache[key] = _rope_pair_norms(W)
    return ctx.cache[key]


def compute_rope_pair_var_Wq(ctx: "HeadContext") -> float:
    """
    Variance of RoPE-pair norms of W_q.

    Groups columns of W_q into consecutive pairs (2k, 2k+1) corresponding
    to RoPE frequency bands and computes the variance of their joint norms.
    High variance indicates that the query projection concentrates energy
    in specific RoPE frequency bands, making the head selective over
    rotational frequencies.
    """
    try:
        pair_norms = _get_cached_rope_pair_norms(ctx, 'rope_pair_norms_Wq', ctx.W_q)
        return float(pair_norms.var().item())
    except Exception as e:
        print(f"Error in compute_rope_pair_var_Wq: {e}")
        return np.nan


def compute_rope_pair_var_Wk(ctx: "HeadContext") -> float:
    """
    Variance of RoPE-pair norms of W_k.

    Symmetric counterpart of compute_rope_pair_var_Wq applied to the key
    projection matrix.
    """
    try:
        pair_norms = _get_cached_rope_pair_norms(ctx, 'rope_pair_norms_Wk', ctx.W_k)
        return float(pair_norms.var().item())
    except Exception as e:
        print(f"Error in compute_rope_pair_var_Wk: {e}")
        return np.nan


def _max_normalized_channel_share(channel_magnitudes: torch.Tensor) -> float:
    """
    Maximum normalized channel share.

    Given non-negative channel magnitudes c_k, compute:
        p_k = c_k / sum_j c_j
    and return max_k p_k.

    Returns a scalar in [0, 1], where higher values indicate a dominant
    channel concentrating most of the mass.
    """
    total = channel_magnitudes.sum()
    if total <= 1e-12:
        return np.nan
    normalized = channel_magnitudes / total
    return float(normalized.max().item())


def _max_over_uniform_channel_share(channel_magnitudes: torch.Tensor) -> float:
    """
    Ratio between maximum normalized channel share and uniform share.

    Given non-negative channel magnitudes c_k, compute:
        p_k = c_k / sum_j c_j,  u = 1 / K
    and return:
        max_k p_k / u = K * max_k p_k

    Returns 1.0 for a perfectly uniform distribution and grows as the top
    channel becomes more dominant.
    """
    total = channel_magnitudes.sum()
    if total <= 1e-12:
        return np.nan
    normalized = channel_magnitudes / total
    K = normalized.shape[0]
    if K == 0:
        return np.nan
    return float((normalized.max() * K).item())


def compute_rope_pair_max_norm_Wq(ctx: "HeadContext") -> float:
    """
    Maximum normalized RoPE-pair channel share of W_q.

    Measures dominance of a single channel after normalization.
    """
    try:
        pair_norms = _get_cached_rope_pair_norms(ctx, 'rope_pair_norms_Wq', ctx.W_q)
        return _max_normalized_channel_share(pair_norms)
    except Exception as e:
        print(f"Error in compute_rope_pair_max_norm_Wq: {e}")
        return np.nan


def compute_rope_pair_max_norm_Wk(ctx: "HeadContext") -> float:
    """
    Maximum normalized RoPE-pair channel share of W_k.

    Measures dominance of a single channel after normalization.
    """
    try:
        pair_norms = _get_cached_rope_pair_norms(ctx, 'rope_pair_norms_Wk', ctx.W_k)
        return _max_normalized_channel_share(pair_norms)
    except Exception as e:
        print(f"Error in compute_rope_pair_max_norm_Wk: {e}")
        return np.nan


def compute_rope_pair_max_uniform_ratio_Wq(ctx: "HeadContext") -> float:
    """
    Ratio of max normalized RoPE-pair share to uniform share for W_q.
    """
    try:
        pair_norms = _get_cached_rope_pair_norms(ctx, 'rope_pair_norms_Wq', ctx.W_q)
        return _max_over_uniform_channel_share(pair_norms)
    except Exception as e:
        print(f"Error in compute_rope_pair_max_uniform_ratio_Wq: {e}")
        return np.nan


def compute_rope_pair_max_uniform_ratio_Wk(ctx: "HeadContext") -> float:
    """
    Ratio of max normalized RoPE-pair share to uniform share for W_k.
    """
    try:
        pair_norms = _get_cached_rope_pair_norms(ctx, 'rope_pair_norms_Wk', ctx.W_k)
        return _max_over_uniform_channel_share(pair_norms)
    except Exception as e:
        print(f"Error in compute_rope_pair_max_uniform_ratio_Wk: {e}")
        return np.nan


def compute_rope_freq_com_Wq(ctx: "HeadContext") -> float:
    """
    RoPE frequency center of mass of W_q.

    Computes the attention-weighted mean frequency index:
        FreqCoM = sum_k(k * c_k) / sum_k(c_k)

    Low values: projection energy concentrated in low-frequency pairs
                -> long-range dependencies, global/sink-like heads.
    High values: energy concentrated in high-frequency pairs
                -> local patterns, near-diagonal/slash heads.
    """
    try:
        pair_norms = _get_cached_rope_pair_norms(ctx, 'rope_pair_norms_Wq', ctx.W_q)
        K = pair_norms.shape[0]
        indices = torch.arange(K, dtype=torch.float32)
        com = float((indices * pair_norms).sum() / (pair_norms.sum() + 1e-12))
        return com
    except Exception as e:
        print(f"Error in compute_rope_freq_com_Wq: {e}")
        return np.nan


def compute_rope_freq_com_Wk(ctx: "HeadContext") -> float:
    """
    RoPE frequency center of mass of W_k.

    Symmetric counterpart of compute_rope_freq_com_Wq applied to W_k.
    """
    try:
        pair_norms = _get_cached_rope_pair_norms(ctx, 'rope_pair_norms_Wk', ctx.W_k)
        K = pair_norms.shape[0]
        indices = torch.arange(K, dtype=torch.float32)
        com = float((indices * pair_norms).sum() / (pair_norms.sum() + 1e-12))
        return com
    except Exception as e:
        print(f"Error in compute_rope_freq_com_Wk: {e}")
        return np.nan


# ==============================================================================
# Attention Map: Diagonal and Shifted Patterns
# ==============================================================================

def _compute_diagonal_mass(ctx: "HeadContext", band_width: int, shift: int = 0) -> float:
    """
    Core implementation of shifted diagonal mass computation.
    
    Args:
        band_width: Width of the band (1 = exactly one diagonal, 3 = target diagonal +/- 1)
        shift: Number of tokens to look back. 
               0 = main diagonal (self-attention)
               1 = first sub-diagonal (attention to previous token)
               d = d-th sub-diagonal (attention to token d steps ago)
    
    Mathematical Definition:
        Mass = sum(A[i,j] for |(i - j) - shift| <= w//2) / sum(A)
    """
    A = ctx.attention_map
    seq_len = A.shape[0]
    half = band_width // 2

    if "diag_dist" not in ctx.cache:
        row = torch.arange(seq_len, device=A.device, dtype=torch.float32).unsqueeze(1)
        col = torch.arange(seq_len, device=A.device, dtype=torch.float32).unsqueeze(0)
        ctx.cache["diag_dist"] = row - col

    dist = ctx.cache["diag_dist"]
    mask = (torch.abs(dist - shift) <= half).float()
    
    total = A.sum()
    if total <= 0:
        return np.nan
    return float((A * mask).sum() / total)


def compute_diagonal_mass_1(ctx: "HeadContext") -> float:
    """Fraction of attention mass on the exact main diagonal (self-attention)."""
    try:
        return _compute_diagonal_mass(ctx, band_width=1, shift=0)
    except Exception as e:
        print(f"Error in compute_diagonal_mass_1: {e}")
        return np.nan


def compute_diagonal_mass_5(ctx: "HeadContext") -> float:
    """Fraction of attention mass within a centered diagonal band of width 5."""
    try:
        return _compute_diagonal_mass(ctx, band_width=5, shift=0)
    except Exception as e:
        print(f"Error in compute_diagonal_mass_5: {e}")
        return np.nan


def compute_shifted_diagonal_mass_1_shift_1(ctx: "HeadContext") -> float:
    """Fraction of attention mass on the exact previous token (shift=1)."""
    try:
        return _compute_diagonal_mass(ctx, band_width=1, shift=1)
    except Exception as e:
        print(f"Error in compute_shifted_diagonal_mass_1_shift_1: {e}")
        return np.nan


def compute_shifted_diagonal_mass_1_shift_2(ctx: "HeadContext") -> float:
    """Fraction of attention mass exactly 2 tokens ago (shift=2)."""
    try:
        return _compute_diagonal_mass(ctx, band_width=1, shift=2)
    except Exception as e:
        print(f"Error in compute_shifted_diagonal_mass_1_shift_2: {e}")
        return np.nan
    
def compute_shifted_diagonal_mass_1_shift_3(ctx: "HeadContext") -> float:
    """Fraction of attention mass exactly 3 tokens ago (shift=3)."""
    try:
        return _compute_diagonal_mass(ctx, band_width=1, shift=3)
    except Exception as e:
        print(f"Error in compute_shifted_diagonal_mass_1_shift_3: {e}")
        return np.nan
    
def compute_shifted_diagonal_mass_1_shift_4(ctx: "HeadContext") -> float:
    """Fraction of attention mass exactly 4 tokens ago (shift=4)."""
    try:
        return _compute_diagonal_mass(ctx, band_width=1, shift=4)
    except Exception as e:
        print(f"Error in compute_shifted_diagonal_mass_1_shift_4: {e}")
        return np.nan

# ==============================================================================
# Attention Map: Sink Mass
# ==============================================================================

def _compute_sink_mass(ctx: "HeadContext", token_pos: int = -1) -> float:
    """
    Core implementation of sink mass metrics.

    Computes per-column sink mass as the average attention received from future
    query positions, normalized by each column's valid causal height:
        Sink_j = mean(A[i, j] for i > j)

    Args:
        token_pos:
            - if >= 0, return Sink_{token_pos}
            - if < 0, return max_j Sink_j
    """
    A = ctx.attention_map
    N = A.shape[0]

    if N < 2:
        return np.nan

    if "sink_per_col" not in ctx.cache:
        row = torch.arange(N, device=A.device).unsqueeze(1)
        col = torch.arange(N, device=A.device).unsqueeze(0)
        mask = (row > col)
        counts = mask.sum(dim=0).float()
        ctx.cache["sink_valid_cols"] = counts > 0
        ctx.cache["sink_per_col"] = (A * mask.float()).sum(dim=0) / counts.clamp(min=1.0)

    sink_per_col = ctx.cache["sink_per_col"]
    valid_cols = ctx.cache["sink_valid_cols"]

    if not torch.any(valid_cols):
        return np.nan

    if token_pos >= 0:
        if token_pos >= N or not bool(valid_cols[token_pos].item()):
            return np.nan
        return float(sink_per_col[token_pos].item())

    return float(sink_per_col[valid_cols].max().item())


def compute_sink_mass_token_0(ctx: "HeadContext") -> float:
    """Average attention received by token 0 (BOS) from all subsequent tokens."""
    try:
        return _compute_sink_mass(ctx, token_pos=0)
    except Exception as e:
        print(f"Error in compute_sink_mass_token_0: {e}")
        return np.nan


def compute_sink_mass_token_1(ctx: "HeadContext") -> float:
    """Average attention received by token 1 from all subsequent tokens."""
    try:
        return _compute_sink_mass(ctx, token_pos=1)
    except Exception as e:
        print(f"Error in compute_sink_mass_token_1: {e}")
        return np.nan


def compute_sink_mass_token_2(ctx: "HeadContext") -> float:
    """Average attention received by token 2 from all subsequent tokens."""
    try:
        return _compute_sink_mass(ctx, token_pos=2)
    except Exception as e:
        print(f"Error in compute_sink_mass_token_2: {e}")
        return np.nan


def compute_sink_mass_token_3(ctx: "HeadContext") -> float:
    """Average attention received by token 3 from all subsequent tokens."""
    try:
        return _compute_sink_mass(ctx, token_pos=3)
    except Exception as e:
        print(f"Error in compute_sink_mass_token_3: {e}")
        return np.nan
    
def compute_sink_mass_token_4(ctx: "HeadContext") -> float:
    """Average attention received by token 4 from all subsequent tokens."""
    try:
        return _compute_sink_mass(ctx, token_pos=4)
    except Exception as e:
        print(f"Error in compute_sink_mass_token_4 {e}")
        return np.nan


def compute_sink_mass_max(ctx: "HeadContext") -> float:
    """
    Maximum per-column sink mass, normalized by each column's valid causal height.

    For each key position j, compute the average attention mass received from all
    future query positions (i > j):
        Sink_j = mean(A[i, j] for i > j)

    Then return the maximum over columns:
        MaxSink = max_j Sink_j
    """
    try:
        return _compute_sink_mass(ctx, token_pos=-1)
    except Exception as e:
        print(f"Error in compute_sink_mass_max: {e}")
        return np.nan


# ==============================================================================
# Attention Map: Entropy and Sparsity
# ==============================================================================


def _get_causal_mask(ctx: "HeadContext") -> torch.Tensor:
    """Causal lower-triangular boolean mask, cached per head."""
    if "causal_mask" not in ctx.cache:
        N = ctx.attention_map.shape[0]
        ctx.cache["causal_mask"] = torch.tril(
            torch.ones(N, N, dtype=torch.bool, device=ctx.attention_map.device)
        )
    return ctx.cache["causal_mask"]

def compute_attention_entropy(ctx: "HeadContext") -> float:
    """
    Shannon entropy of causal attention rows — vectorized.
    H = mean_i( -sum_j A[i,j] * log(A[i,j]) )  for j <= i only.
    """
    try:
        A = ctx.attention_map.float()
        N = A.shape[0]
        if N < 2:
            return np.nan
        mask = _get_causal_mask(ctx)
        A_masked = (A * mask).clamp(min=1e-12)
        # log(clamp) su celle fuori dalla maschera produce valori negativi
        # ma li azzeriamo moltiplicando di nuovo per mask
        row_entropies = -(A * torch.log(A_masked) * mask).sum(dim=1)  # [N]
        return float(row_entropies.mean().item())
    except Exception as e:
        print(f"Error in compute_attention_entropy: {e}")
        return np.nan


def compute_attention_gini(ctx: "HeadContext") -> float:
    """
    Gini coefficient of the attention weight distribution.

    Measures pure inequality (sparsity), independently of position.
    Gini = 0: perfectly uniform. Gini = 1: single-position attention.

    Mathematical Definition:
        G = (2 * sum(i * w_i) / (n * sum(w))) - (n+1)/n
        where w_i are sorted in ascending order.
    """
    try:
        mask = _get_causal_mask(ctx)
        w, _ = torch.sort(ctx.attention_map[mask].flatten())
        n = w.shape[0]
        idx = torch.arange(1, n + 1, dtype=torch.float32)
        gini = (2.0 * (idx * w).sum() / (n * w.sum() + 1e-12)) - (n + 1.0) / n
        return float(gini.item())
    except Exception as e:
        print(f"Error in compute_attention_gini: {e}")
        return np.nan




def compute_attention_row_var_weighted(ctx: "HeadContext") -> float:
    """
    Degrees-of-freedom weighted mean of per-row variances of A.
    Rows are weighted by their number of valid causal elements (i+1),
    correcting for instability of variance estimates in early rows.
    """
    try:
        A = ctx.attention_map.float()  # [N, N]
        N = A.shape[0]
        if N < 2:
            return np.nan
        weights, variances = [], []
        mask = _get_causal_mask(ctx).clone()
        mask[0] = False  # skip row 0
        counts = mask.sum(dim=1).float()  # (N,)
        means = (A * mask).sum(dim=1) / counts.clamp(min=1)
        sq_diff = ((A - means.unsqueeze(1)) ** 2) * mask
        variances = sq_diff.sum(dim=1) / counts.clamp(min=1)
        weights = counts
        return float((weights[1:] * variances[1:]).sum() / weights[1:].sum())

    except Exception as e:
        print(f"Error in compute_attention_row_var_weighted: {e}")
        return np.nan



# ==============================================================================
# Attention Map: Structural Metrics
# ==============================================================================

def compute_look_back(ctx: "HeadContext") -> float:
    """
    Normalised look-back distance (Kumar et al., 2024).
    LB_norm = (1/N) * sum_i sum_j [(i-j) * A_ij].

    """
    
    try:
        A = ctx.attention_map.float()
        N = A.shape[0]
        row = torch.arange(N, dtype=torch.float32, device=A.device).unsqueeze(1)
        col = torch.arange(N, dtype=torch.float32, device=A.device).unsqueeze(0)
        distances = (row - col).clamp(min=0)
        # Lookback normalizzato per riga
        # max distanza per la riga i è i. Se i=0, distanza max è 1 (per evitare div/0)
        max_distances = row.clamp(min=1) 
        norm_distances = distances / max_distances
        row_lookbacks = (A * norm_distances).sum(dim=1)
        return float(row_lookbacks.mean().item())
    except Exception as e:
        print(f"Error in compute_look_back: {e}")
        return np.nan


# ==============================================================================
# Attention Map: Rank
# ==============================================================================

def compute_effective_rank_A(ctx: "HeadContext") -> float:
    try:    return _get_cached_rank(ctx, 'svd_A', ctx.attention_map)['effective_rank']
    except Exception as e: print(f"Error in compute_effective_rank_A: {e}"); return np.nan

def compute_r95_A(ctx: "HeadContext") -> float:
    try:    return float(_get_cached_rank(ctx, 'svd_A', ctx.attention_map)['r95'])
    except Exception as e: print(f"Error in compute_r95_A: {e}"); return np.nan


# ==============================================================================
# Feature Registry
# ==============================================================================

FEATURE_REGISTRY: Dict[str, Callable] = {

    # --- Weight Matrix Ranks (W_q, W_k, W_v) ---
    "effective_rank_Wq":            compute_effective_rank_Wq,
    "r95_Wq":                       compute_r95_Wq,
    "effective_rank_Wk":            compute_effective_rank_Wk,
    "r95_Wk":                       compute_r95_Wk,
    "effective_rank_Wv":            compute_effective_rank_Wv,
    "r95_Wv":                       compute_r95_Wv,

    # --- Hidden State Rank (H) ---
    "effective_rank_H":             compute_effective_rank_H,
    "r95_H":                        compute_r95_H,

    # --- Projected Q and K Ranks ---
    "effective_rank_Q":             compute_effective_rank_Q,
    "r95_Q":                        compute_r95_Q,
    "effective_rank_K":             compute_effective_rank_K,
    "r95_K":                        compute_r95_K,

    # --- Temporal Similarity ---
    "q_sim_consecutive":            compute_q_sim_consecutive,
    "k_sim_consecutive":            compute_k_sim_consecutive,

    # --- SVD Alignment (H vs projections) ---
    "svd_alignment_H_Wq":          compute_svd_alignment_H_Wq,
    "svd_alignment_H_Wk":          compute_svd_alignment_H_Wk,

    # --- RoPE-aware QK alignment ---
    "compute_WqRWk_alignment_delta_0": compute_WqRWk_alignment_delta_0,
    "compute_WqRWk_alignment_delta_1": compute_WqRWk_alignment_delta_1,
    "compute_WqRWk_alignment_delta_2": compute_WqRWk_alignment_delta_2,
    "compute_WqRWk_alignment_delta_3": compute_WqRWk_alignment_delta_3,
    "compute_WqRWk_alignment_delta_4": compute_WqRWk_alignment_delta_4,

    # --- RMSNorm and Channel Structure ---
    "rmsnorm_gamma_norm":           compute_rmsnorm_gamma_norm,

    # Channel spread and dominance over RoPE pairs
    "rope_pair_var_Wq":             compute_rope_pair_var_Wq,
    "rope_pair_var_Wk":             compute_rope_pair_var_Wk,
    "rope_pair_max_norm_Wq":        compute_rope_pair_max_norm_Wq,
    "rope_pair_max_norm_Wk":        compute_rope_pair_max_norm_Wk,
    "rope_pair_max_ratio_Wq":       compute_rope_pair_max_uniform_ratio_Wq,
    "rope_pair_max_ratio_Wk":       compute_rope_pair_max_uniform_ratio_Wk,
    
    "rope_freq_com_Wq":             compute_rope_freq_com_Wq,
    "rope_freq_com_Wk":             compute_rope_freq_com_Wk,

    # --- Attention Map: Diagonal ---
    "diagonal_mass_1":              compute_diagonal_mass_1,
    "diagonal_mass_5":              compute_diagonal_mass_5,
    "diagonal_mass_1_shifted_1":    compute_shifted_diagonal_mass_1_shift_1,
    "diagonal_mass_1_shifted_2":    compute_shifted_diagonal_mass_1_shift_2,
    "diagonal_mass_1_shifted_3":    compute_shifted_diagonal_mass_1_shift_3,
    "diagonal_mass_1_shifted_4":    compute_shifted_diagonal_mass_1_shift_4,

    # --- Attention Map: Sink ---
    "sink_mass_token_0":            compute_sink_mass_token_0,
    "sink_mass_token_1":            compute_sink_mass_token_1,
    "sink_mass_token_2":            compute_sink_mass_token_2,
    "sink_mass_token_3":            compute_sink_mass_token_3,
    "sink_mass_token_4":            compute_sink_mass_token_4,
    "sink_mass_max":                compute_sink_mass_max,

    # --- Attention Map: Entropy and Variance ---
    "attention_entropy":            compute_attention_entropy,
    "attention_gini":               compute_attention_gini,
    "attention_row_var_weighted":   compute_attention_row_var_weighted,

    # --- Attention Map: Structural ---
    "look_back":                    compute_look_back,

    # --- Attention Map: Rank ---
    "effective_rank_A":             compute_effective_rank_A,
    "r95_A":                        compute_r95_A,
}


# ==============================================================================
# Get All Features Function
# ==============================================================================

def get_all_features(ctx: "HeadContext") -> Dict[str, float]:
    """
    Compute all registered features for a given HeadContext.
    Iterates through FEATURE_REGISTRY with graceful failure handling.
    Returns a flat dictionary of scalar floats (np.nan on failure).
    """
    results: Dict[str, float] = {}
    for name, func in FEATURE_REGISTRY.items():
        try:
            results[name] = func(ctx)
        except Exception as e:
            print(f"Warning: feature '{name}' failed with: {e}")
            results[name] = np.nan
    return results