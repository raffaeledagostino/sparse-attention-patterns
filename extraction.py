"""Attention extraction and RoPE analysis utilities."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


# =====================================================================
# Standard extraction (post-softmax, with RoPE)
# =====================================================================

def extract_attention_standard(model: Any, inputs: dict[str, Any]) -> list[torch.Tensor]:
    """
    Extract post-softmax attention weights using the model's built-in
    attention output (includes RoPE).

    Returns
    -------
    list[torch.Tensor]
        One tensor per layer, each of shape (1, num_heads, T, T).
    """
    model.config.use_cache = False
    with torch.no_grad():
        out = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=False,
        )
    return [a.detach().cpu() for a in out.attentions]


# =====================================================================
# Pre-softmax extraction (with RoPE)
# =====================================================================

def extract_presoftmax_with_rope(model: Any, inputs: dict[str, Any]) -> list[torch.Tensor]:
    """
    Capture attention logits immediately before softmax while keeping the
    model's standard attention path (including RoPE).

    Returns
    -------
    list[torch.Tensor]
        Pre-softmax logits per layer — each shape (1, num_heads, T, T).
    """
    captured_logits = []
    original_softmax = F.softmax

    def _capturing_softmax(input, dim=None, _stacklevel=3, dtype=None):
        if input.dim() == 4 and input.shape[-1] == input.shape[-2]:
            captured_logits.append(input.detach().cpu())
        return original_softmax(input, dim=dim, dtype=dtype)

    F.softmax = _capturing_softmax
    try:
        model.config.use_cache = False
        with torch.no_grad():
            model(
                **inputs,
                output_attentions=False,
                output_hidden_states=False,
            )
    finally:
        F.softmax = original_softmax

    num_layers = len(model.model.layers)
    if len(captured_logits) < num_layers:
        raise RuntimeError(
            f"Captured {len(captured_logits)} pre-softmax tensors, expected at least {num_layers}."
        )

    return captured_logits[-num_layers:]


def _build_position_ids(inputs: dict[str, Any], seq_len: int) -> torch.Tensor:
    """Return position IDs consistent with the model's default cache counting."""
    if "position_ids" in inputs and inputs["position_ids"] is not None:
        return inputs["position_ids"]
    device = inputs["input_ids"].device
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)


def _project_query_key(attn_layer: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_shape = (*hidden_states.shape[:-1], -1, attn_layer.head_dim)
    query_dtype = attn_layer.q_proj.weight.dtype
    key_dtype = attn_layer.k_proj.weight.dtype
    common_dtype = query_dtype if query_dtype == key_dtype else torch.promote_types(query_dtype, key_dtype)
    hidden_states = hidden_states.to(common_dtype)
    query_states = attn_layer.q_norm(attn_layer.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = attn_layer.k_norm(attn_layer.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    return query_states.to(torch.float32), key_states.to(torch.float32)


def extract_layer_query_key_states(
    model: Any,
    inputs: dict[str, Any],
    layer_idx: int,
    hidden_states: tuple[torch.Tensor, ...] | None = None,
    apply_rope: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return query and key tensors for a single layer.
    
    Parameters
    ----------
    model : PreTrainedModel
        The transformer model.
    inputs : dict
        Tokenized inputs.
    layer_idx : int
        Layer index.
    hidden_states : tuple, optional
        Pre-computed hidden states from the model.
    apply_rope : bool, default=True
        If True, apply RoPE (Post-RoPE states for attention computation).
        If False, return Pre-RoPE states (for semantic/temporal analysis).
    
    Returns
    -------
    query_states, key_states : torch.Tensor
        Shape (batch, num_heads, seq_len, head_dim) each.
        If apply_rope=False: Pre-RoPE states (semantic similarity preserved).
        If apply_rope=True: Post-RoPE states (ready for attention computation).
    """
    if hidden_states is None:
        model.config.use_cache = False
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states

    if layer_idx >= len(hidden_states):
        raise IndexError("layer_idx is out of range for the available hidden states")

    layer_input = hidden_states[layer_idx]
    attn_layer = model.model.layers[layer_idx].self_attn

    query_states, key_states = _project_query_key(attn_layer, layer_input)
    
    if apply_rope:
        position_ids = _build_position_ids(inputs, layer_input.shape[1])
        cos, sin = model.model.rotary_emb(hidden_states[0], position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # Always repeat KV for Grouped Query Attention, even for Pre-RoPE analysis
    # (GQA repeats the same semantic content to match Q head count)
    key_states = repeat_kv(key_states, attn_layer.num_key_value_groups)

    return query_states.detach().cpu().float(), key_states.detach().cpu().float()


def compute_adjacent_cosines(matrix: np.ndarray) -> np.ndarray:
    """Compute adjacent-row cosine similarities.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix of shape ``(n_rows, d)``.

    Returns
    -------
    np.ndarray
        Cosine similarities for adjacent rows with shape ``(n_rows - 1,)``.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D with shape (n_rows, d)")

    states = matrix
    norms = np.linalg.norm(states, axis=-1)
    numerator = np.sum(states[:-1] * states[1:], axis=-1)
    denominator = norms[:-1] * norms[1:]
    denominator = np.where(denominator == 0, 1e-6, denominator)
    cos = numerator / denominator
    return np.clip(cos, -1.0, 1.0)


def compute_head_adjacent_cosines(
    model: Any,
    inputs: dict[str, Any],
    layer_idx: int,
    head_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return adjacent cosine similarities for queries and keys of a specific head.
    
    IMPORTANT: Uses Pre-RoPE states to preserve semantic/temporal similarity.
    RoPE's high-frequency rotations would artificially destroy temporal similarity.
    
    Parameters
    ----------
    model : PreTrainedModel
    inputs : dict
        Tokenized inputs.
    layer_idx : int
    head_idx : int
    
    Returns
    -------
    query_cos, key_cos : np.ndarray
        Cosine similarities cos(q_i, q_{i+1}) and cos(k_i, k_{i+1}).
        Shape: (seq_len - 1,) each.
    """
    # Use Pre-RoPE states for temporal cosine similarity (TAPPA paper framework)
    query_states, key_states = extract_layer_query_key_states(model, inputs, layer_idx, apply_rope=False)
    num_heads = query_states.shape[1]
    if head_idx >= num_heads:
        raise IndexError(
            f"head_idx ({head_idx}) is out of range (num_attention_heads={num_heads})"
        )

    query_vals = query_states[0, head_idx].numpy()
    key_vals = key_states[0, head_idx].numpy()

    return compute_adjacent_cosines(query_vals), compute_adjacent_cosines(key_vals)


def compute_rank_metrics(tensor_pre_rope: torch.Tensor | np.ndarray) -> dict[str, Any]:
    """
    Compute rank-related metrics for a Pre-RoPE matrix (Q or K for one head).
    
    Reproduces the rank analysis from "Demystifying the Slash Pattern in Attention".
    
    Parameters
    ----------
    tensor_pre_rope : torch.Tensor or np.ndarray
        Pre-RoPE matrix of shape (seq_len, head_dim).
    
    Returns
    -------
    dict
        Contains:
        - 'singular_values': np.ndarray of shape (min(seq_len, head_dim),)
        - 'rank1_dominance': float
            σ₁² / Σσᵢ² — fraction of total variance captured by first singular value
        - 'effective_rank': float
            Shannon entropy-based continuous measure of rank:
            exp(-Σ pᵢ log pᵢ) where pᵢ = σᵢ² / Σσⱼ²
    """
    if isinstance(tensor_pre_rope, torch.Tensor):
        matrix = tensor_pre_rope.cpu().numpy()
    else:
        matrix = tensor_pre_rope
    
    # Compute SVD: U @ diag(S) @ Vh ≈ matrix
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    
    # Singular values squared (eigenvalues of matrix @ matrix.T)
    S_squared = S ** 2
    total_variance = np.sum(S_squared)
    
    # Rank-1 dominance: how much variance is captured by the first singular vector
    rank1_dominance = S_squared[0] / total_variance if total_variance > 0 else 0.0
    
    # Effective rank via Shannon entropy
    # Normalize squared singular values to probabilities
    p = S_squared / total_variance if total_variance > 0 else np.ones_like(S_squared) / len(S_squared)
    # Entropy: -Σ pᵢ log pᵢ
    # Filter out zeros to avoid log(0)
    p_nonzero = p[p > 1e-12]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    effective_rank = np.exp(entropy)
    
    # R_0.95: minimum number of singular values needed to explain 95% of variance
    # Calculate cumulative normalized singular values squared
    cumsum = np.cumsum(S_squared) / total_variance if total_variance > 0 else np.cumsum(S_squared)
    r_095 = int(np.argmax(cumsum >= 0.95) + 1) if np.any(cumsum >= 0.95) else len(S)
    
    return {
        'singular_values': S,
        'rank1_dominance': rank1_dominance,
        'effective_rank': effective_rank,
        'r_095': r_095,
    }


def compute_rope_channel_utilization(
    q_head_pre: np.ndarray,
    k_head_pre: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute RoPE-channel utilization statistics for Q and K.

    Parameters
    ----------
    q_head_pre : np.ndarray
        Pre-RoPE query states with shape ``(seq_len, head_dim)``.
    k_head_pre : np.ndarray
        Pre-RoPE key states with shape ``(seq_len, head_dim)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(q_avg_pct, k_avg_pct, q_avg_mag, k_avg_mag)`` where percentage arrays
        describe average per-channel relative usage and magnitude arrays describe
        average absolute per-channel norms.
    """
    seq_len, head_dim = q_head_pre.shape
    num_channels = head_dim // 2

    q_mag = np.zeros((seq_len, num_channels))
    k_mag = np.zeros((seq_len, num_channels))

    for channel_idx in range(num_channels):
        q_mag[:, channel_idx] = np.sqrt(
            q_head_pre[:, 2 * channel_idx] ** 2 + q_head_pre[:, 2 * channel_idx + 1] ** 2
        )
        k_mag[:, channel_idx] = np.sqrt(
            k_head_pre[:, 2 * channel_idx] ** 2 + k_head_pre[:, 2 * channel_idx + 1] ** 2
        )

    q_total_mag = np.sum(q_mag, axis=1, keepdims=True) + 1e-10
    k_total_mag = np.sum(k_mag, axis=1, keepdims=True) + 1e-10
    q_pct = (q_mag / q_total_mag) * 100.0
    k_pct = (k_mag / k_total_mag) * 100.0

    q_avg_pct = np.mean(q_pct, axis=0)
    k_avg_pct = np.mean(k_pct, axis=0)
    q_avg_mag = np.mean(q_mag, axis=0)
    k_avg_mag = np.mean(k_mag, axis=0)
    return q_avg_pct, k_avg_pct, q_avg_mag, k_avg_mag


def apply_rope(x: np.ndarray, base: float) -> np.ndarray:
    """Apply RoPE rotation to a 2D matrix.

    Parameters
    ----------
    x : np.ndarray
        Input matrix of shape ``(seq_len, head_dim)``.
    base : float
        RoPE base (theta denominator).

    Returns
    -------
    np.ndarray
        Rotated matrix with the same shape as ``x``.
    """
    seq_len, dim = x.shape
    theta = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    freqs = np.outer(np.arange(seq_len), theta)
    x_pairs = x.reshape(seq_len, -1, 2)
    x0, x1 = x_pairs[:, :, 0], x_pairs[:, :, 1]
    xr0 = x0 * np.cos(freqs) - x1 * np.sin(freqs)
    xr1 = x0 * np.sin(freqs) + x1 * np.cos(freqs)
    return np.stack([xr0, xr1], axis=-1).reshape(seq_len, dim)


def extract_logit_features(
    q_head_pre: np.ndarray,
    k_head_pre: np.ndarray,
    logits: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract aligned cosine and logit features for correlation analysis.

    Parameters
    ----------
    q_head_pre : np.ndarray
        Pre-RoPE query states with shape ``(seq_len, head_dim)``.
    k_head_pre : np.ndarray
        Pre-RoPE key states with shape ``(seq_len, head_dim)``.
    logits : np.ndarray
        Attention logits matrix with shape ``(seq_len, seq_len)``.

    Returns
    -------
    dict[str, np.ndarray]
        Feature arrays containing adjacent cosine signals and corresponding
        near-diagonal logit values.

    Notes
    -----
    This preserves the notebook's feature alignment intent by pairing
    ``cos(q_i, q_{i+1})`` and ``cos(k_i, k_{i+1})`` with first sub-diagonal logits.
    """
    q_cos = compute_adjacent_cosines(q_head_pre)
    k_cos = compute_adjacent_cosines(k_head_pre)

    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("logits must be a square 2D matrix")

    sub_diag = np.diagonal(logits, offset=-1)
    n = min(len(q_cos), len(k_cos), len(sub_diag))
    return {
        "q_adjacent_cos": q_cos[:n],
        "k_adjacent_cos": k_cos[:n],
        "logit_subdiag": sub_diag[:n],
    }


def get_save_path(category: str, filename: str) -> Path:
    """Return an output path and create the category directory if missing.

    Parameters
    ----------
    category : str
        Output category directory under ``outputs/``.
    filename : str
        Target file name.

    Returns
    -------
    Path
        Full path ``outputs/<category>/<filename>``.
    """
    project_root = Path(__file__).resolve().parents[1]
    category_dir = project_root / "outputs" / category
    category_dir.mkdir(parents=True, exist_ok=True)
    return category_dir / filename


def find_slash_dominant_heads(
    attention_weights_tensor: torch.Tensor | list[torch.Tensor],
    max_lag: int = 5,
    threshold: float = 0.2,
) -> list[dict[str, float | int]]:
    """
    Detect Slash-Dominant Heads (SDHs) using average sub-diagonal attention scores.

    For each lag Δ in [0, max_lag], computes:
        score(Δ) = mean(attn[i, i-Δ]) over valid i

    Parameters
    ----------
    attention_weights_tensor : torch.Tensor or list[torch.Tensor]
        Attention probabilities with shape (num_layers, batch, num_heads, seq_len, seq_len),
        or a list of per-layer tensors each shaped (batch, num_heads, seq_len, seq_len).
    max_lag : int, default=5
        Maximum lower sub-diagonal offset Δ to evaluate.
    threshold : float, default=0.2
        Minimum average slash score to classify a head as SDH at lag Δ.

    Returns
    -------
    list[dict]
        Each record contains:
        - layer_id
        - head_id
        - lag
        - slash_score
    """
    if isinstance(attention_weights_tensor, list):
        if len(attention_weights_tensor) == 0:
            return []
        attn = torch.stack(attention_weights_tensor, dim=0)
    else:
        attn = attention_weights_tensor

    if not isinstance(attn, torch.Tensor):
        raise TypeError("attention_weights_tensor must be a torch.Tensor or list[torch.Tensor]")

    if attn.dim() != 5:
        raise ValueError(
            "Expected attention tensor shape (num_layers, batch, num_heads, seq_len, seq_len)"
        )

    num_layers, batch_size, num_heads, seq_len, _ = attn.shape
    if batch_size < 1:
        return []

    # Use first batch as requested (batch_size assumed 1 in this workflow)
    matrices = attn[:, 0].detach().float()  # (L, H, T, T)

    max_valid_lag = min(max_lag, seq_len - 1)
    results = []

    for delta in range(max_valid_lag + 1):
        # Vectorized extraction across all layers/heads for a fixed lag Δ
        diagonal_vals = torch.diagonal(matrices, offset=-delta, dim1=-2, dim2=-1)  # (L, H, T-Δ)
        slash_scores = torch.nanmean(diagonal_vals, dim=-1)  # (L, H)

        sdh_mask = slash_scores >= threshold
        sdh_indices = torch.nonzero(sdh_mask, as_tuple=False)

        for layer_id, head_id in sdh_indices.tolist():
            results.append(
                {
                    "layer_id": int(layer_id),
                    "head_id": int(head_id),
                    "lag": int(delta),
                    "slash_score": float(slash_scores[layer_id, head_id].item()),
                }
            )

    return results
