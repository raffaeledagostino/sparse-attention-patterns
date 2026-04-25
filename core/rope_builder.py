
"""
rope_builders.py — Model-specific RoPE rotation matrix constructors.

Defines R_delta ∈ R^{d_head × d_head} for three architectures:

  Model                               rope_type     rope_theta  QK-norm
  ─────────────────────────────────────────────────────────────────────
  Qwen/Qwen3-4B                       standard      1e6         YES
  mistralai/Mistral-7B-Instruct-v0.3  standard      1e4         NO
  meta-llama/Llama-3.1-8B-Instruct    llama3        5e5         NO

Exported API
────────────
  build_rope_rotation_qwen3(delta, d_head, rope_theta)
  build_rope_rotation_mistral(delta, d_head, rope_theta)
  build_rope_rotation_llama31(delta, d_head, rope_theta, ...)
  get_rope_builder(model_name)          -> Callable[(delta, d_head) -> Tensor]
  get_rope_builder_from_config(config)  -> Callable[(delta, d_head) -> Tensor]
  get_inv_freq(model_name, d_head)      -> Tensor  (diagnostic / visualisation)
"""

import math
from typing import Callable, Dict

import torch


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Internal primitives
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_rotation_from_inv_freq(delta: int, inv_freq: torch.Tensor) -> torch.Tensor:
    """
    Build the block-diagonal RoPE rotation matrix from precomputed frequencies.

    Mathematical definition:

        R_delta = bigoplus_{m=0}^{d_h/2 - 1}
                  [[cos(delta * theta_m), -sin(delta * theta_m)],
                   [sin(delta * theta_m),  cos(delta * theta_m)]]

    where theta_m = inv_freq[m].

    Args:
        delta    : relative position offset (integer >= 0)
        inv_freq : (d_head/2,) angular frequencies, dtype float32, on CPU

    Returns:
        R_delta : (d_head, d_head) block-diagonal matrix, float32, CPU
    """
    half   = inv_freq.shape[0]
    d_head = half * 2
    angles = inv_freq * float(delta)   # (half,)
    cos_a  = angles.cos()
    sin_a  = angles.sin()

    idx = torch.arange(half, dtype=torch.long) * 2   # [0, 2, 4, ..., d_head-2]
    R   = torch.zeros(d_head, d_head, dtype=torch.float32)
    R[idx,     idx    ] =  cos_a
    R[idx,     idx + 1] = -sin_a
    R[idx + 1, idx    ] =  sin_a
    R[idx + 1, idx + 1] =  cos_a
    return R


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Inverse-frequency constructors (one per RoPE variant)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _inv_freq_standard(d_head: int, rope_theta: float) -> torch.Tensor:
    """
    Standard RoPE inverse frequencies (Qwen3, Mistral, original RoFormer).

        theta_m = rope_theta^{-2m / d_head},   m = 0, ..., d_head/2 - 1

    Geometric spectrum: theta_0 = 1  (highest freq)  down to
                        theta_{d/2-1} = rope_theta^{-1}  (lowest freq).

    With rope_theta=1e4 (Mistral): theta_8 ≈ 0.316  →  Delta* ≈ 3-6
    With rope_theta=1e6 (Qwen3):   theta_8 ≈ 0.063  →  Delta* >> 64 (inaccessible)
    """
    m = torch.arange(0, d_head, 2, dtype=torch.float32)
    return 1.0 / (rope_theta ** (m / d_head))


def _inv_freq_llama31(
    d_head: int,
    rope_theta: float      = 5e5,
    factor: float          = 8.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    orig_max_pos: int      = 8192,
) -> torch.Tensor:
    """
    LLaMA 3.1 non-uniform frequency scaling ("llama3" rope_type).

    Partitions the d_head/2 channels into three regimes using wavelength
    lambda_m = 2*pi / theta_m as discriminant:

      High-frequency  (lambda_m  <  lambda_high = orig_max_pos / high_freq_factor):
          theta_m_scaled = theta_m            (no scaling, keep original freq)

      Low-frequency   (lambda_m  >  lambda_low  = orig_max_pos / low_freq_factor):
          theta_m_scaled = theta_m / factor   (compress by 8x, push Delta* further)

      Transition      (lambda_high <= lambda_m <= lambda_low):
          smooth = (orig_max_pos/lambda_m - low_freq_factor)
                   / (high_freq_factor - low_freq_factor)
          theta_m_scaled = (1 - smooth)*theta_m/factor + smooth*theta_m

    Default LLaMA-3.1 params: rope_theta=5e5, factor=8, low_ff=1, high_ff=4,
    orig_max_pos=8192  →  lambda_high=2048, lambda_low=8192.

    Channel classification for d_head=128, rope_theta=5e5:
      m = 0..9   : high-frequency, unscaled  → contributes to Delta* ≈ 5-7
      m = 10..41 : transition band
      m = 42..63 : low-frequency, ÷8         → Delta* >> 64 (inaccessible)

    CRITICAL: Using standard inv_freq with rope_theta=5e5 for LLaMA 3.1 produces
    WRONG rotation matrices for all low-frequency channels (m ≥ 42), biasing
    subspace_overlap and trace_norm toward zero at large delta.
    """
    base   = _inv_freq_standard(d_head, rope_theta)  # (half,)
    half   = base.shape[0]

    lambda_high = orig_max_pos / high_freq_factor   # 2048.0
    lambda_low  = orig_max_pos / low_freq_factor    # 8192.0

    scaled = torch.empty(half, dtype=torch.float32)
    for i in range(half):
        freq    = base[i].item()
        wavelen = 2.0 * math.pi / (freq + 1e-30)

        if wavelen < lambda_high:
            scaled[i] = freq
        elif wavelen > lambda_low:
            scaled[i] = freq / factor
        else:
            # Linear interpolation in the transition band
            smooth    = (orig_max_pos / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            scaled[i] = (1.0 - smooth) * freq / factor + smooth * freq

    return scaled


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public rotation builders (one per model)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_rope_rotation_qwen3(
    delta: int,
    d_head: int = 128,
    rope_theta: float = 1_000_000.0,
) -> torch.Tensor:
    """
    RoPE rotation matrix for Qwen/Qwen3-4B.

    Spec:
      rope_theta = 1e6  (standard RoPE, no scaling)
      QK-norm    = YES  (applied BEFORE RoPE in the forward pass)

    QK-norm note: normalisation operates on the projected Q/K vectors before
    they are rotated by R_delta. It does NOT change R_delta — the rotation
    matrix is identical to the standard formula. What changes is that the
    input norms ||q_t|| = ||k_t|| = 1 (controlled by RMSNorm gamma), so the
    amplitude weights A_m in the channel decomposition are fixed to 1,
    making constructive interference driven purely by phase. With rope_theta=1e6
    the phase condition theta_{m*} * Delta* = -psi_{m*} cannot be satisfied
    within a 64-token window, so attention collapses to delta=0 (R_0 = I).

    Args:
        delta      : relative position offset
        d_head     : head dimension (default 128)
        rope_theta : RoPE base (default 1e6)

    Returns:
        R_delta : (d_head, d_head) rotation matrix, float32, CPU
    """
    return _build_rotation_from_inv_freq(
        delta, _inv_freq_standard(d_head, rope_theta)
    )


def build_rope_rotation_mistral(
    delta: int,
    d_head: int = 128,
    rope_theta: float = 10_000.0,
) -> torch.Tensor:
    """
    RoPE rotation matrix for mistralai/Mistral-7B-Instruct-v0.3.

    Spec:
      rope_theta = 1e4  (standard RoPE, no scaling)
      QK-norm    = NO
      GQA        = 32Q : 8KV  (kv_idx = q_idx // 4)
      Sliding window = 4096 (inactive for seq_len <= 64)

    With rope_theta=1e4, the dominant channel m*=8 has theta_8 ≈ 0.316.
    For a typical phase offset psi_{m*} ≈ -1 rad:
        Delta* ≈ 1/0.316 ≈ 3.2  →  expected peak at Delta = 3-6.

    This is the ONLY model of the three where subspace_overlap and trace_norm
    should show a clear peak at a non-zero, accessible offset.

    Args:
        delta      : relative position offset
        d_head     : head dimension (default 128)
        rope_theta : RoPE base (default 1e4)

    Returns:
        R_delta : (d_head, d_head) rotation matrix, float32, CPU
    """
    return _build_rotation_from_inv_freq(
        delta, _inv_freq_standard(d_head, rope_theta)
    )


def build_rope_rotation_llama31(
    delta: int,
    d_head: int             = 128,
    rope_theta: float       = 500_000.0,
    factor: float           = 8.0,
    low_freq_factor: float  = 1.0,
    high_freq_factor: float = 4.0,
    orig_max_pos: int       = 8192,
) -> torch.Tensor:
    """
    RoPE rotation matrix for meta-llama/Llama-3.1-8B-Instruct.

    Spec:
      rope_theta   = 5e5
      rope_type    = "llama3"  (non-uniform scaling)
      QK-norm      = NO
      GQA          = 32Q : 8KV

    Channel regime breakdown (d_head=128, rope_theta=5e5):
      m =  0.. 9 : high-freq, unscaled,  theta_m ≈ 0.12-1.0
                   → contributes to accessible Delta* ≈ 5-7
      m = 10..41 : transition band, partially scaled
      m = 42..63 : low-freq, divided by 8, theta_m << 0.01
                   → Delta* >> 64 tokens (no contribution to accessible offsets)

    Expected behaviour: correlation peak at Delta ≈ 5-7, but weaker than Mistral
    because ~half the channels contribute incoherently (bimodal regime split).

    Args:
        delta            : relative position offset
        d_head           : head dimension (default 128)
        rope_theta       : base frequency (default 5e5)
        factor           : low-freq compression factor (default 8.0)
        low_freq_factor  : orig_max_pos / lambda_low  (default 1.0)
        high_freq_factor : orig_max_pos / lambda_high (default 4.0)
        orig_max_pos     : original context length (default 8192)

    Returns:
        R_delta : (d_head, d_head) rotation matrix, float32, CPU
    """
    return _build_rotation_from_inv_freq(
        delta,
        _inv_freq_llama31(
            d_head, rope_theta, factor,
            low_freq_factor, high_freq_factor, orig_max_pos,
        ),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factory: by model name
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Registry: model-name prefix → builder function with the correct defaults bound
_ROPE_REGISTRY: Dict[str, Callable[[int, int], torch.Tensor]] = {}


def _register():
    import functools
    _ROPE_REGISTRY["Qwen/Qwen3"]                    = build_rope_rotation_qwen3
    _ROPE_REGISTRY["Qwen/Qwen2"]                    = functools.partial(
        build_rope_rotation_qwen3, rope_theta=1e4
    )
    _ROPE_REGISTRY["mistralai/Mistral"]             = build_rope_rotation_mistral
    _ROPE_REGISTRY["mistralai/Mixtral"]             = build_rope_rotation_mistral
    _ROPE_REGISTRY["meta-llama/Llama-3.1"]          = build_rope_rotation_llama31
    _ROPE_REGISTRY["meta-llama/Meta-Llama-3.1"]     = build_rope_rotation_llama31

_register()


def get_rope_builder(model_name: str) -> Callable[[int, int], torch.Tensor]:
    """
    Return a RoPE builder for a model name with all defaults pre-bound.

    Returned callable signature:
        builder(delta: int, d_head: int) -> torch.Tensor  # (d_head, d_head)

    Example:
        build_R = get_rope_builder("meta-llama/Llama-3.1-8B-Instruct")
        R5 = build_R(delta=5, d_head=128)

    Raises:
        ValueError if no matching prefix is found.
    """
    for prefix, builder in _ROPE_REGISTRY.items():
        if model_name.startswith(prefix):
            return builder
    raise ValueError(
        f"No RoPE builder registered for '{model_name}'."

        f"Known prefixes: {list(_ROPE_REGISTRY.keys())}"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factory: from HuggingFace model.config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_rope_builder_from_config(model_config) -> Callable[[int, int], torch.Tensor]:
    """
    Return a RoPE builder by reading directly from a HuggingFace model config.

    Reads model_config.rope_theta and model_config.rope_scaling.
    Dispatches to _inv_freq_llama31 if rope_scaling.rope_type == "llama3",
    otherwise falls back to _inv_freq_standard.

    This is the RECOMMENDED entry point when running inside a Kaggle notebook
    where the model is already loaded, because it reads all hyperparameters
    directly from the config without hardcoding model names.

    Example:
        build_R = get_rope_builder_from_config(model.config)
        R3 = build_R(delta=3, d_head=128)
        R3 = build_R(delta=3, d_head=model.config.head_dim)
    """
    rope_theta   = float(getattr(model_config, "rope_theta", 1e4))
    rope_scaling = getattr(model_config, "rope_scaling", None)

    if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
        _factor   = float(rope_scaling.get("factor", 8.0))
        _low_ff   = float(rope_scaling.get("low_freq_factor", 1.0))
        _high_ff  = float(rope_scaling.get("high_freq_factor", 4.0))
        _orig_max = int(rope_scaling.get("original_max_position_embeddings", 8192))

        def builder(delta: int, d_head: int) -> torch.Tensor:
            return _build_rotation_from_inv_freq(
                delta,
                _inv_freq_llama31(d_head, rope_theta, _factor, _low_ff, _high_ff, _orig_max),
            )
    else:
        def builder(delta: int, d_head: int) -> torch.Tensor:
            return _build_rotation_from_inv_freq(
                delta, _inv_freq_standard(d_head, rope_theta)
            )

    return builder


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Diagnostic: frequency spectrum inspection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_inv_freq(model_name: str, d_head: int = 128) -> torch.Tensor:
    """
    Return the (d_head/2,) inverse frequency tensor for a given model.

    Useful for visualising the channel frequency spectrum and verifying the
    LLaMA 3.1 scaling regime boundaries. Returns float32 CPU tensor.

    Example:
        import pandas as pd
        freqs_ll = get_inv_freq("meta-llama/Llama-3.1-8B-Instruct")
        freqs_ms = get_inv_freq("mistralai/Mistral-7B-Instruct-v0.3")
        freqs_qw = get_inv_freq("Qwen/Qwen3-4B")
        # Plot freqs_ll vs freqs_ms to see the bimodal split in LLaMA 3.1
    """
    if model_name.startswith("meta-llama/Llama-3.1") or model_name.startswith("meta-llama/Meta-Llama-3.1"):
        return _inv_freq_llama31(d_head)
    elif model_name.startswith("Qwen/Qwen3"):
        return _inv_freq_standard(d_head, 1e6)
    elif model_name.startswith("mistralai/Mistral"):
        return _inv_freq_standard(d_head, 1e4)
    else:
        raise ValueError(f"Unknown model for get_inv_freq: {model_name}")


def get_delta_star_profile(
    model_name: str,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    d_head: int = 128,
    delta_max: int = 50,
) -> torch.Tensor:
    """
    Compute the constructive interference profile F(delta) for a single head.

    F(delta) = sum_m A_m * cos(psi_m + delta * theta_m)

    where:
      A_m   = ||u_1^{(m)}|| * ||v_1^{(m)}||     (amplitude in RoPE channel m)
      psi_m = angle(u_1^{(m)}) - angle(v_1^{(m)})  (relative phase)
      theta_m = inv_freq[m]

    Uses the rank-1 approximation (leading singular vectors only).
    The argmax of F gives the predicted preferred offset Delta* for this head.

    Args:
        model_name : used to select the correct inv_freq variant
        W_q        : (d_head, d_model) query projection weight (CPU float32)
        W_k        : (d_head, d_model) key projection weight (CPU float32)
        d_head     : head dimension
        delta_max  : maximum offset to evaluate

    Returns:
        F : (delta_max+1,) tensor of interference values
    """
    inv_freq = get_inv_freq(model_name, d_head)                  # (half,)
    half     = d_head // 2

    M0      = W_q.float().cpu() @ W_k.float().cpu().T            # (d_head, d_head)
    U, _, Vh = torch.linalg.svd(M0, full_matrices=False)
    u1 = U[:, 0]   # leading left sing. vector
    v1 = Vh[0, :]  # leading right sing. vector

    A_m   = torch.zeros(half)
    psi_m = torch.zeros(half)
    for m in range(half):
        u_m = u1[[2*m, 2*m+1]]
        v_m = v1[[2*m, 2*m+1]]
        A_m[m]   = u_m.norm() * v_m.norm()
        angle_u  = torch.atan2(u_m[1], u_m[0])
        angle_v  = torch.atan2(v_m[1], v_m[0])
        psi_m[m] = (angle_u - angle_v).item()

    deltas = torch.arange(0, delta_max + 1, dtype=torch.float32)
    F = torch.stack([
        (A_m * torch.cos(psi_m + d * inv_freq)).sum()
        for d in deltas
    ])
    return F