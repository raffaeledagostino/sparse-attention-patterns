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
# Single SVD Entry Point
# ==============================================================================

def _svdvals_cpu(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute singular values of a matrix on CPU (MPS-safe).

    Args:
        matrix: 2D float tensor of shape (m, n).

    Returns:
        1D float tensor of singular values in descending order.
    """
    m = matrix.detach().cpu().float()
    try:
        return torch.linalg.svdvals(m)
    except Exception:
        _, s, _ = torch.svd(m)
        return s


def _compute_rank_metrics(matrix: torch.Tensor) -> Dict[str, float]:
    """
    Compute effective rank and R_95 from a 2D matrix via a single SVD call.

    Effective rank is defined as exp(H(p)), where H is the Shannon entropy
    of the normalized singular value distribution. R_95 is the minimum number
    of singular values whose cumulative mass reaches 95% of the total.

    Args:
        matrix: 2D tensor of shape (m, n).

    Returns:
        dict with keys:
            'effective_rank': float, exp(Shannon entropy of normalized singular values)
            'r95':            int,   minimum k s.t. sum(s[:k]) / sum(s) >= 0.95
    """
    s = _svdvals_cpu(matrix)
    total = s.sum() + 1e-12
    probs = s / total

    # Effective Rank: exp(H(p))
    p_nz = probs[probs > 1e-12]
    entropy = -torch.sum(p_nz * torch.log(p_nz))
    effective_rank = float(torch.exp(entropy).item())

    # R_95: smallest k such that cumulative mass >= 0.95
    cumsum = torch.cumsum(probs, dim=0)
    r95 = int((cumsum < 0.95).sum().item()) + 1

    return {"effective_rank": effective_rank, "r95": r95}


def _get_cached_rank(ctx: "HeadContext", key: str, matrix: torch.Tensor) -> Dict[str, float]:
    """
    Retrieve rank metrics from ctx.cache, computing them only once per (layer, head).

    Args:
        ctx:    HeadContext instance carrying the shared cache dict.
        key:    Cache key string (e.g., 'rank_Q', 'rank_Wq').
        matrix: The 2D tensor to decompose if the cache is cold.

    Returns:
        dict with 'effective_rank' and 'r95'.
    """
    if key not in ctx.cache:
        ctx.cache[key] = _compute_rank_metrics(matrix)
    return ctx.cache[key]


# ==============================================================================
# Rank of Weight Matrices (W_q, W_k, W_v)
# ==============================================================================

def compute_effective_rank_Wq(ctx: "HeadContext") -> float:
    """Effective rank of the weight matrix W_q for this head."""
    try:
        return _get_cached_rank(ctx, 'rank_Wq', ctx.W_q)['effective_rank']
    except Exception as e:
        print(f"Error in compute_effective_rank_Wq: {e}")
        return np.nan


def compute_r95_Wq(ctx: "HeadContext") -> float:
    """R_95% of the weight matrix W_q for this head."""
    try:
        return float(_get_cached_rank(ctx, 'rank_Wq', ctx.W_q)['r95'])
    except Exception as e:
        print(f"Error in compute_r95_Wq: {e}")
        return np.nan


def compute_effective_rank_Wk(ctx: "HeadContext") -> float:
    """Effective rank of the weight matrix W_k for this head."""
    try:
        return _get_cached_rank(ctx, 'rank_Wk', ctx.W_k)['effective_rank']
    except Exception as e:
        print(f"Error in compute_effective_rank_Wk: {e}")
        return np.nan


def compute_r95_Wk(ctx: "HeadContext") -> float:
    """R_95% of the weight matrix W_k for this head."""
    try:
        return float(_get_cached_rank(ctx, 'rank_Wk', ctx.W_k)['r95'])
    except Exception as e:
        print(f"Error in compute_r95_Wk: {e}")
        return np.nan


def compute_effective_rank_Wv(ctx: "HeadContext") -> float:
    """Effective rank of the weight matrix W_v for this head."""
    try:
        return _get_cached_rank(ctx, 'rank_Wv', ctx.W_v)['effective_rank']
    except Exception as e:
        print(f"Error in compute_effective_rank_Wv: {e}")
        return np.nan


def compute_r95_Wv(ctx: "HeadContext") -> float:
    """R_95% of the weight matrix W_v for this head."""
    try:
        return float(_get_cached_rank(ctx, 'rank_Wv', ctx.W_v)['r95'])
    except Exception as e:
        print(f"Error in compute_r95_Wv: {e}")
        return np.nan


# ==============================================================================
# Rank of Hidden States H
# ==============================================================================

def compute_effective_rank_H(ctx: "HeadContext") -> float:
    """
    Effective rank of the input hidden state matrix H.

    H has shape (seq_len, d_model) and is shared across all heads in the same
    layer. The cache ensures the SVD is computed only once per layer.
    """
    try:
        return _get_cached_rank(ctx, 'rank_H', ctx.H_input)['effective_rank']
    except Exception as e:
        print(f"Error in compute_effective_rank_H: {e}")
        return np.nan


def compute_r95_H(ctx: "HeadContext") -> float:
    """R_95% of the input hidden state matrix H."""
    try:
        return float(_get_cached_rank(ctx, 'rank_H', ctx.H_input)['r95'])
    except Exception as e:
        print(f"Error in compute_r95_H: {e}")
        return np.nan


# ==============================================================================
# Rank of Projected Q and K
# ==============================================================================

def compute_effective_rank_Q(ctx: "HeadContext") -> float:
    """
    Effective rank of the projected Query matrix Q = H @ W_q^T.

    Mathematical Definition:
        s = SVD(Q),  p_i = s_i / sum(s)
        effective_rank = exp(-sum(p_i * log(p_i)))
    """
    try:
        return _get_cached_rank(ctx, 'rank_Q', ctx.Q)['effective_rank']
    except Exception as e:
        print(f"Error in compute_effective_rank_Q: {e}")
        return np.nan


def compute_r95_Q(ctx: "HeadContext") -> float:
    """R_95% of the projected Query matrix Q."""
    try:
        return float(_get_cached_rank(ctx, 'rank_Q', ctx.Q)['r95'])
    except Exception as e:
        print(f"Error in compute_r95_Q: {e}")
        return np.nan


def compute_effective_rank_K(ctx: "HeadContext") -> float:
    """Effective rank of the projected Key matrix K = H @ W_k^T."""
    try:
        return _get_cached_rank(ctx, 'rank_K', ctx.K)['effective_rank']
    except Exception as e:
        print(f"Error in compute_effective_rank_K: {e}")
        return np.nan


def compute_r95_K(ctx: "HeadContext") -> float:
    """R_95% of the projected Key matrix K."""
    try:
        return float(_get_cached_rank(ctx, 'rank_K', ctx.K)['r95'])
    except Exception as e:
        print(f"Error in compute_r95_K: {e}")
        return np.nan


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
def _top2_right_singular_vectors(matrix: torch.Tensor) -> torch.Tensor:
    """
    Return the top-2 right singular vectors of a matrix.
    Both H and W live in R^{d_model} as their column space,
    so right singular vectors are geometrically comparable across matrices.

    Returns:
        Tensor of shape (2, d_model) — top-2 rows of V^T.
    """
    m = matrix.detach().cpu().float()
    try:
        _, _, Vt = torch.linalg.svd(m, full_matrices=False)
    except Exception:
        _, _, Vt = torch.svd(m)
        Vt = Vt.T
    return Vt[:2]  # shape: (2, d_model)

def _principal_angle_alignment(V1: torch.Tensor, V2: torch.Tensor) -> float:
    """
    Mean cosine of principal angles between subspaces spanned by rows of V1, V2.
    V1: (k, d), V2: (k, d) — rows are orthonormal (from SVD).
    sigma_i(V1 @ V2.T) = cos(theta_i), i=1..k.
    1.0 = identical subspaces, 0.0 = orthogonal subspaces.
    """
    G = V1.cpu() @ V2.cpu().T          # (k, k) Gram matrix
    cos_angles = torch.linalg.svdvals(G.float())   # [0, 1]
    return float(cos_angles.mean().item())

def compute_svd_alignment_H_Wq(ctx: "HeadContext") -> float:
    """
    Mean cosine of principal angles between the top-2 right singular
    subspaces of H and W_q in the shared input space R^{d_model}.
    Uses principal angles (optimal matching) instead of positional pairing.
    """
    try:
        V_H  = _top2_right_singular_vectors(ctx.H_input)  # (2, d_model)
        V_Wq = _top2_right_singular_vectors(ctx.W_q)      # (2, d_model)
        return _principal_angle_alignment(V_H, V_Wq)
    except Exception as e:
        print(f"Error in compute_svd_alignment_H_Wq: {e}")
        return np.nan

def compute_svd_alignment_H_Wk(ctx: "HeadContext") -> float:
    """
    Mean cosine of principal angles between the top-2 right singular
    subspaces of H and W_k in the shared input space R^{d_model}.
    """
    try:
        V_H  = _top2_right_singular_vectors(ctx.H_input)  # (2, d_model)
        V_Wk = _top2_right_singular_vectors(ctx.W_k)      # (2, d_model)
        return _principal_angle_alignment(V_H, V_Wk)
    except Exception as e:
        print(f"Error in compute_svd_alignment_H_Wk: {e}")
        return np.nan

def _get_cached_WqWk_svd(ctx: "HeadContext") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cached full SVD of the head interaction matrix M = W_q @ W_k^T in R^{d_h x d_h}.

    M_{ij} = (e_i^T W_q)(W_k^T e_j) captures the bilinear interaction
    between query and key projected channels, independently of the input.
    """
    if 'svd_WqWk' not in ctx.cache:
        Wq = ctx.W_q.detach().cpu().float()  # [d_h, d_model]
        Wk = ctx.W_k.detach().cpu().float()  # [d_h, d_model]
        M = Wq @ Wk.T                        # [d_h, d_h]
        try:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        except Exception:
            U, S, V = torch.svd(M)
            Vh = V.T   # normalizza subito alla convenzione linalg
        ctx.cache['svd_WqWk'] = (U, S, Vh)
    return ctx.cache['svd_WqWk']


def compute_WqWk_svd_alignment(ctx: "HeadContext") -> float:
    """
    Singular-value-weighted cosine similarity between left and right
    singular vectors of the head interaction matrix M = W_q W_k^T.

    Following Zhang et al. (NeurIPS 2024, arXiv:2405.14880), the attention
    score decomposes into singular modes: each mode n contributes
    sigma_n * (x_i^T u_n)(v_n^T x_j). A high weighted alignment means
    the head's query and key sides 'look for the same features' in the
    leading singular modes.

    rho = sum_n  (sigma_n / sum_m sigma_m) * <u_n, v_n>

    Note: u_n and v_n are already unit-norm (columns of U and V from SVD),
    so <u_n, v_n> is directly their cosine similarity.
    Input-independent: computed once per head over the weight matrices.
    """
    try:
        U, S, Vh = _get_cached_WqWk_svd(ctx)
        V = Vh.T  # [d_h, r]
        cos_sim = (U * V).sum(dim=0)        # [r]
        weights = S / (S.sum() + 1e-12)
        return float((weights * cos_sim).sum().item())
    except Exception as e:
        print(f"Error in compute_WqWk_svd_alignment: {e}")
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
        Tensor of shape (d_model // 2,) with one norm per frequency pair.
    """
    m = W.detach().cpu().float()
    
    row_norms_sq = (m ** 2).sum(dim=1)           # (d_h,)
    pair_norms_sq = row_norms_sq.view(-1, 2).sum(dim=1)  # (d_h//2,)
    return pair_norms_sq.sqrt()                           # (d_model//2,)


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
        pair_norms = _rope_pair_norms(ctx.W_q)
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
        pair_norms = _rope_pair_norms(ctx.W_k)
        return float(pair_norms.var().item())
    except Exception as e:
        print(f"Error in compute_rope_pair_var_Wk: {e}")
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
        pair_norms = _rope_pair_norms(ctx.W_q)
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
        pair_norms = _rope_pair_norms(ctx.W_k)
        K = pair_norms.shape[0]
        indices = torch.arange(K, dtype=torch.float32)
        com = float((indices * pair_norms).sum() / (pair_norms.sum() + 1e-12))
        return com
    except Exception as e:
        print(f"Error in compute_rope_freq_com_Wk: {e}")
        return np.nan


# ==============================================================================
# Attention Map: Diagonal Pattern
# ==============================================================================

def _compute_diagonal_mass(ctx: "HeadContext", band_width: int) -> float:
    """
    Core implementation of diagonal mass computation.

    Mathematical Definition:
        DiagMass_w = sum(A[i,j] for |i-j| <= w//2) / sum(A)
    """
    A = ctx.attention_map
    seq_len = A.shape[0]
    half = band_width // 2
    row = torch.arange(seq_len, device=A.device, dtype=torch.float32).unsqueeze(1)
    col = torch.arange(seq_len, device=A.device, dtype=torch.float32).unsqueeze(0)
    mask = (torch.abs(row - col) <= half).float()
    total = A.sum()
    if total <= 0:
        return np.nan
    return float((A * mask).sum() / total)

# ==============================================================================
# Attention Map: Diagonal and Shifted Patterns
# ==============================================================================

def _compute_shifted_diagonal_mass(ctx: "HeadContext", band_width: int, shift: int = 0) -> float:
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
    
    row = torch.arange(seq_len, device=A.device, dtype=torch.float32).unsqueeze(1)
    col = torch.arange(seq_len, device=A.device, dtype=torch.float32).unsqueeze(0)
    
    # Distance is (row - col). We center the band around 'shift'
    mask = (torch.abs((row - col) - shift) <= half).float()
    
    total = A.sum()
    if total <= 0:
        return np.nan
    return float((A * mask).sum() / total)


def compute_diagonal_mass_1(ctx: "HeadContext") -> float:
    """Fraction of attention mass on the exact main diagonal (self-attention)."""
    try:
        return _compute_shifted_diagonal_mass(ctx, band_width=1, shift=0)
    except Exception as e:
        print(f"Error in compute_diagonal_mass_1: {e}")
        return np.nan


def compute_diagonal_mass_5(ctx: "HeadContext") -> float:
    """Fraction of attention mass within a centered diagonal band of width 5."""
    try:
        return _compute_shifted_diagonal_mass(ctx, band_width=5, shift=0)
    except Exception as e:
        print(f"Error in compute_diagonal_mass_5: {e}")
        return np.nan


def compute_shifted_diagonal_mass_1_shift_1(ctx: "HeadContext") -> float:
    """Fraction of attention mass on the exact previous token (shift=1)."""
    try:
        return _compute_shifted_diagonal_mass(ctx, band_width=1, shift=1)
    except Exception as e:
        print(f"Error in compute_shifted_diagonal_mass_1_shift_1: {e}")
        return np.nan


def compute_shifted_diagonal_mass_1_shift_2(ctx: "HeadContext") -> float:
    """Fraction of attention mass exactly 2 tokens ago (shift=2)."""
    try:
        return _compute_shifted_diagonal_mass(ctx, band_width=1, shift=2)
    except Exception as e:
        print(f"Error in compute_shifted_diagonal_mass_1_shift_2: {e}")
        return np.nan
    
def compute_shifted_diagonal_mass_1_shift_3(ctx: "HeadContext") -> float:
    """Fraction of attention mass exactly 3 tokens ago (shift=3)."""
    try:
        return _compute_shifted_diagonal_mass(ctx, band_width=1, shift=3)
    except Exception as e:
        print(f"Error in compute_shifted_diagonal_mass_1_shift_3: {e}")
        return np.nan
    
def compute_shifted_diagonal_mass_1_shift_4(ctx: "HeadContext") -> float:
    """Fraction of attention mass exactly 4 tokens ago (shift=4)."""
    try:
        return _compute_shifted_diagonal_mass(ctx, band_width=1, shift=4)
    except Exception as e:
        print(f"Error in compute_shifted_diagonal_mass_1_shift_4: {e}")
        return np.nan

# ==============================================================================
# Attention Map: Per-Token Sink Mass
# ==============================================================================

def _compute_single_token_sink_mass(ctx: "HeadContext", token_pos: int) -> float:
    """
    Core implementation of per-token sink mass.

    Computes the average attention received by a single token at position
    `token_pos`, averaged over all query positions that come strictly after it
    (causal mask: i > token_pos only, excluding self-attention on the diagonal).

    Args:
        token_pos: Absolute position of the candidate sink token (0-indexed).

    Mathematical Definition:
        Sink_j = mean(A[i, j] for i > j)  =  mean(A[j+1:, j])
    """
    A = ctx.attention_map
    N = A.shape[0]

    # Need at least one query position after token_pos
    if N <= token_pos + 1:
        return np.nan

    # Column j, rows strictly below the diagonal (causal queries only)
    sink_column = A[token_pos + 1:, token_pos]  # shape: (N - token_pos - 1,)
    return float(sink_column.mean().item())


def compute_sink_mass_token_0(ctx: "HeadContext") -> float:
    """Average attention received by token 0 (BOS) from all subsequent tokens."""
    try:
        return _compute_single_token_sink_mass(ctx, token_pos=0)
    except Exception as e:
        print(f"Error in compute_sink_mass_token_0: {e}")
        return np.nan


def compute_sink_mass_token_1(ctx: "HeadContext") -> float:
    """Average attention received by token 1 from all subsequent tokens."""
    try:
        return _compute_single_token_sink_mass(ctx, token_pos=1)
    except Exception as e:
        print(f"Error in compute_sink_mass_token_1: {e}")
        return np.nan


def compute_sink_mass_token_2(ctx: "HeadContext") -> float:
    """Average attention received by token 2 from all subsequent tokens."""
    try:
        return _compute_single_token_sink_mass(ctx, token_pos=2)
    except Exception as e:
        print(f"Error in compute_sink_mass_token_2: {e}")
        return np.nan


def compute_sink_mass_token_3(ctx: "HeadContext") -> float:
    """Average attention received by token 3 from all subsequent tokens."""
    try:
        return _compute_single_token_sink_mass(ctx, token_pos=3)
    except Exception as e:
        print(f"Error in compute_sink_mass_token_3: {e}")
        return np.nan
    
def compute_sink_mass_token_4(ctx: "HeadContext") -> float:
    """Average attention received by token 3 from all subsequent tokens."""
    try:
        return _compute_single_token_sink_mass(ctx, token_pos=4)
    except Exception as e:
        print(f"Error in compute_sink_mass_token_4 {e}")
        return np.nan


# ==============================================================================
# Attention Map: Entropy and Sparsity
# ==============================================================================

def compute_attention_entropy(ctx: "HeadContext") -> float:
    """
    Shannon entropy of the full post-softmax attention map.

    High entropy indicates dispersed (dense) attention.
    Low entropy indicates concentrated (sparse) attention.

    Mathematical Definition:
        H = -sum_{i,j} A[i,j] * log(A[i,j])  for A[i,j] > 0
    """
    try:
        A = ctx.attention_map.flatten()
        A = A[A > 1e-12]
        if len(A) == 0:
            return np.nan
        entropy = -torch.sum(A * torch.log(A))
        return float(entropy.item())
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
        w, _ = torch.sort(ctx.attention_map.flatten().cpu().float())
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
        mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=A.device))
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
        A = ctx.attention_map
        N = A.shape[0]
        if N < 2:
            return np.nan
        row = torch.arange(1, N + 1, dtype=torch.float32, device=A.device).unsqueeze(1)
        col = torch.arange(1, N + 1, dtype=torch.float32, device=A.device).unsqueeze(0)
        weights = (row - col).clamp(min=0) 
        return float((A * weights).sum() / N)
    except Exception as e:
        print(f"Error in compute_look_back: {e}")
        return np.nan


# ==============================================================================
# Attention Map: Rank
# ==============================================================================

def compute_effective_rank_A(ctx: "HeadContext") -> float:
    """Effective rank of the post-softmax attention matrix A."""
    try:
        return _get_cached_rank(ctx, 'rank_A', ctx.attention_map)['effective_rank']
    except Exception as e:
        print(f"Error in compute_effective_rank_A: {e}")
        return np.nan


def compute_r95_A(ctx: "HeadContext") -> float:
    """R_95% of the post-softmax attention matrix A."""
    try:
        return float(_get_cached_rank(ctx, 'rank_A', ctx.attention_map)['r95'])
    except Exception as e:
        print(f"Error in compute_r95_A: {e}")
        return np.nan


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
    "WqWk_svd_alignment":          compute_WqWk_svd_alignment,

    # --- RMSNorm and Channel Structure ---
    "rmsnorm_gamma_norm":           compute_rmsnorm_gamma_norm,
    "rope_pair_var_Wq":          compute_rope_pair_var_Wq,
    "rope_pair_var_Wk":          compute_rope_pair_var_Wk,
    "rope_freq_com_Wq":             compute_rope_freq_com_Wq,
    "rope_freq_com_Wk":             compute_rope_freq_com_Wk,

    # --- Attention Map: Diagonal ---
    "diagonal_mass_1":              compute_diagonal_mass_1,
    "diagonal_mass_5":              compute_diagonal_mass_5,
    "diagonal_mass_1_shifted_1":    compute_shifted_diagonal_mass_1_shift_1,
    "diagonal_mass_1_shifted_2":    compute_shifted_diagonal_mass_1_shift_2,
    "diagonal_mass_1_shifted_3":    compute_shifted_diagonal_mass_1_shift_3,
    "diagonal_mass_1_shifted_4":    compute_shifted_diagonal_mass_1_shift_4,

    # --- Attention Map: Sink (per-token, independent) ---
    "sink_mass_token_0":            compute_sink_mass_token_0,
    "sink_mass_token_1":            compute_sink_mass_token_1,
    "sink_mass_token_2":            compute_sink_mass_token_2,
    "sink_mass_token_3":            compute_sink_mass_token_3,
    "sink_mass_token_4":            compute_sink_mass_token_4,

    # --- Attention Map: Entropy and Sparsity ---
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

    Iterates through FEATURE_REGISTRY, runs each function with graceful
    failure handling, and returns a flat dictionary of scalar floats.

    Args:
        ctx: HeadContext instance for a single (layer, head) pair.

    Returns:
        Dict[str, float]: Feature name to scalar value. np.nan on failure.
    """
    results: Dict[str, float] = {}
    for name, func in FEATURE_REGISTRY.items():
        try:
            results[name] = func(ctx)
        except Exception as e:
            print(f"Warning: feature '{name}' failed with: {e}")
            results[name] = np.nan
    return results