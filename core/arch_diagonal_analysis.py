"""Architectural-vs-diagonal attention analysis scaffold.

This module defines the analysis interface for comparing architecture-derived
metrics from attention weights against empirical diagonal attention mass.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from datasets import load_dataset
from tqdm import tqdm

from config import (
    ATTN_IMPLEMENTATION,
    DATASET_CONFIG,
    DATASET_NAME,
    DATASET_SPLIT,
    DATASET_TEXT_COLUMN,
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_MPS,
    MODEL_NAME,
    MIN_CHARS,
    RANDOM_SEED,
    TARGET_TOKENS,
)
from core.features_library import _economy_svd, _to_svd_tensor, build_rope_rotation
from data.persistence import save_results
from rope_builder import get_rope_builder_from_config


# ── Constants ─────────────────────────────────────────────────────────────────

DELTA_MAX  = 20       # max sub-diagonal offset to analyse
N_PROMPTS  = 30       # number of wikitext prompts for empirical mass
K_SUBSPACE = 4        # number of leading singular vectors for subspace overlap
H_Q        = 32       # number of query heads  (Qwen3-4B)
H_KV       = 8        # number of KV heads     (Qwen3-4B, GQA)
HEAD_DIM   = 128      # d_h per head           (Qwen3-4B)
ROPE_THETA = 1e6      # RoPE base frequency    (Qwen3-4B)

ARCH_METRICS_PATH = Path("data/arch_metrics.parquet")
DIAG_MASS_PATH    = Path("data/diag_mass.parquet")
CORRELATION_PATH  = Path("data/correlation_arch_vs_diag.parquet")

ARCH_PRIMARY_KEY = ["model_name", "layer_idx", "head_idx", "delta"]
MASS_PRIMARY_KEY = ["model_name", "prompt_id", "layer_idx", "head_idx", "delta"]
CORR_PRIMARY_KEY = ["model_name", "layer_idx", "delta", "metric_name"]

# All SVD must run on CPU float32 — Apple Silicon + Kaggle T4 safe.
_GQA_RATIO = H_Q // H_KV  # 4: each KV head serves 4 query heads


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(func, *args, fallback=np.nan):
    """Call func(*args), return fallback on any exception."""
    try:
        return func(*args)
    except Exception as e:
        print(f"[_safe] {func.__name__}: {e}")
        return fallback


def _spectral_entropy(S: torch.Tensor) -> float:
    p = S / (S.sum() + 1e-12)
    p_nz = p[p > 1e-12]
    return float(-torch.sum(p_nz * torch.log(p_nz)).item())


def _subspace_overlap(U_K: torch.Tensor, V_K: torch.Tensor, R: torch.Tensor) -> float:
    """SO = (1/K) * sum_c |u_c^T @ R @ v_c|.

    Args:
        U_K : (HEAD_DIM, K) — leading K left singular vectors (columns).
        V_K : (HEAD_DIM, K) — leading K right singular vectors (columns, = Vh[:K].T).
        R   : (HEAD_DIM, HEAD_DIM) — RoPE rotation matrix at offset delta.
    """
    RV   = R @ V_K              # (HEAD_DIM, K)
    dots = (U_K * RV).sum(0)   # (K,)  — elementwise dot product per column
    return float(dots.abs().mean().item())


def _trace_norm(M0: torch.Tensor, R: torch.Tensor, M0_fro: float) -> float:
    """tau = |tr(M0 @ R.T)| / ||M0||_F."""
    trace_val = float(torch.trace(M0 @ R.T).item())
    return abs(trace_val) / (M0_fro + 1e-12)


# ── Part 1 ────────────────────────────────────────────────────────────────────

def compute_arch_metrics(model, model_name: str) -> pd.DataFrame:
    """Compute 4 architectural metrics per (layer, head, delta) from weights only.

    Iterates over all transformer layers. For each query head, extracts the
    per-head W_q and W_k slices (respecting GQA mapping), computes the
    interaction matrix M0 = W_q @ W_k.T once, caches its SVD, then evaluates
    the four metrics for every delta in [0, DELTA_MAX]:

      - sigma1           : leading singular value of M(delta) = M0 @ R_delta.T
                           Theoretically delta-invariant; serves as diagnostic.
      - spectral_entropy : -sum_i p_i log p_i  where p_i = sigma_i / sum(sigma)
                           Also delta-invariant; measures rank concentration.
      - subspace_overlap : (1/K) * sum_{c=1}^{K} |u_c^T @ R_delta @ v_c|
                           Measures how much R_delta aligns left and right
                           singular subspaces. Peaks at the preferred offset.
      - trace_norm       : |tr(M0 @ R_delta.T)| / ||M0||_F
                           Normalised trace; independent measure of same effect.

    Output schema:
        ["model_name","layer_idx","head_idx","delta",
         "sigma1","spectral_entropy","subspace_overlap","trace_norm"]

    Saves to ARCH_METRICS_PATH via save_results (idempotent merge).
    """
    layers    = model.model.layers
    n_layers  = len(layers)
    rope_theta = float(getattr(model.config, "rope_theta", ROPE_THETA))
    results   = []

    for layer_idx in tqdm(range(n_layers), desc="arch_metrics — layers"):
        attn          = layers[layer_idx].self_attn
        q_proj_weight = attn.q_proj.weight.detach()   # (H_Q*HEAD_DIM, d_model)
        k_proj_weight = attn.k_proj.weight.detach()   # (H_KV*HEAD_DIM, d_model)

        for head_idx in range(H_Q):
            kv_idx = head_idx // _GQA_RATIO

            W_q = _to_svd_tensor(
                q_proj_weight[head_idx * HEAD_DIM:(head_idx + 1) * HEAD_DIM, :]
            )  # (HEAD_DIM, d_model)
            W_k = _to_svd_tensor(
                k_proj_weight[kv_idx * HEAD_DIM:(kv_idx + 1) * HEAD_DIM, :]
            )  # (HEAD_DIM, d_model)

            M0 = W_q @ W_k.T  # (HEAD_DIM, HEAD_DIM)

            # ── Cache SVD of M0 once per head ──────────────────────────────
            try:
                U, S, Vh = _economy_svd(M0)
            except Exception as e:
                print(f"[arch] SVD failed layer={layer_idx} head={head_idx}: {e}")
                for delta in range(DELTA_MAX + 1):
                    results.append(dict(model_name=model_name, layer_idx=layer_idx,
                                        head_idx=head_idx, delta=delta,
                                        sigma1=np.nan, spectral_entropy=np.nan,
                                        subspace_overlap=np.nan, trace_norm=np.nan))
                continue

            # delta-invariant quantities (compute once, reuse across delta loop)
            sigma1           = _safe(lambda: float(S[0].item()))
            spectral_entropy = _safe(_spectral_entropy, S)
            M0_fro           = _safe(lambda: float(torch.norm(M0, p="fro").item()))

            U_K = U[:, :K_SUBSPACE]   # (HEAD_DIM, K)
            V_K = Vh[:K_SUBSPACE, :].T  # (HEAD_DIM, K) — right sing. vecs as columns

            build_R = get_rope_builder_from_config(model.config)  # chiamato una volta

            # ── Delta loop ─────────────────────────────────────────────────
            for delta in range(DELTA_MAX + 1):
                try:
                    build_R = get_rope_builder_from_config(model.config)  # chiamato una volta
                    R = build_R(delta=delta, d_head=HEAD_DIM)
                    so  = _safe(_subspace_overlap, U_K, V_K, R)
                    tau = _safe(_trace_norm, M0, R, M0_fro)
                except Exception as e:
                    print(f"[arch] R failed delta={delta}: {e}")
                    so, tau = np.nan, np.nan

                results.append(dict(
                    model_name=model_name, layer_idx=layer_idx,
                    head_idx=head_idx, delta=delta,
                    sigma1=sigma1, spectral_entropy=spectral_entropy,
                    subspace_overlap=so, trace_norm=tau,
                ))

    return save_results(results, ARCH_METRICS_PATH, ARCH_PRIMARY_KEY)


# ── Part 2 ────────────────────────────────────────────────────────────────────

def compute_diag_mass(model, tokenizer, model_name: str) -> pd.DataFrame:
    """Compute empirical diagonal mass per (prompt, layer, head, delta).

    Streams N_PROMPTS samples from wikitext-103-raw-v1 train split,
    tokenizes each to exactly TARGET_TOKENS tokens, runs a causal forward
    pass with output_attentions=True, and extracts:

        dm(h, delta) = mean_{t=delta}^{T-1} A[t, t-delta]

    which equals np.diag(A, -delta).mean() — the mean attention weight
    assigned to tokens exactly `delta` positions in the past.

    GQA note: with attn_implementation="eager", Qwen3 returns attention
    tensors of shape (1, H_Q, T, T), already broadcast over KV heads.

    Output schema:
        ["model_name","prompt_id","layer_idx","head_idx","delta","diag_mass"]

    Saves to DIAG_MASS_PATH via save_results (idempotent merge).
    """
    dataset = load_dataset(
        DATASET_NAME, DATASET_CONFIG,
        split=DATASET_SPLIT, streaming=True,
        trust_remote_code=True,
    )

    results       = []
    prompt_count  = 0
    device        = next(model.parameters()).device

    pbar = tqdm(total=N_PROMPTS, desc="diag_mass — prompts")

    for sample in dataset:
        text = sample[DATASET_TEXT_COLUMN]
        if len(text) < MIN_CHARS:
            continue

        enc = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=TARGET_TOKENS,
        )
        if enc["input_ids"].shape[1] < TARGET_TOKENS:
            continue

        input_ids = enc["input_ids"][:, :TARGET_TOKENS].to(device)
        prompt_id = f"prompt_{prompt_count}"

        with torch.no_grad():
            out = model(input_ids, output_attentions=True)

        # out.attentions: tuple[n_layers] of (1, H_Q, T, T)
        for layer_idx, attn_tensor in enumerate(out.attentions):
            A_all = attn_tensor[0].float().cpu()  # (H_Q, T, T)

            for head_idx in range(H_Q):
                A = A_all[head_idx]  # (T, T)

                for delta in range(DELTA_MAX + 1):
                    try:
                        # torch.diag(A, -delta): sub-diagonal at offset delta
                        # shape: (T - delta,) — exactly the t>=delta slice
                        diag_vals = torch.diag(A, -delta)
                        dm = float(diag_vals.mean().item()) if diag_vals.numel() > 0 else np.nan
                    except Exception:
                        dm = np.nan

                    results.append(dict(
                        model_name=model_name, prompt_id=prompt_id,
                        layer_idx=layer_idx, head_idx=head_idx,
                        delta=delta, diag_mass=dm,
                    ))

        prompt_count += 1
        pbar.update(1)
        if prompt_count >= N_PROMPTS:
            break

    pbar.close()
    return save_results(results, DIAG_MASS_PATH, MASS_PRIMARY_KEY)


# ── Part 3 ────────────────────────────────────────────────────────────────────

def compute_correlations(model_name: str) -> pd.DataFrame:
    """Compute Pearson/Spearman correlations between arch metrics and diag mass.

    For each (layer_idx, delta) and each architectural metric in
    [subspace_overlap, trace_norm, sigma1, spectral_entropy]:

        x = metric values across H_Q heads (from arch_metrics.parquet)
        y = mean diag_mass across N_PROMPTS per head (from diag_mass.parquet)

        pearson_r               : Pearson correlation on (x, y)
        spearman_rho            : Spearman rank correlation on (x, y)
        pearson_std_over_prompts: std of Pearson(x, y_n) across individual prompts
                                  — stability diagnostic; low std = input-independent effect

    All NaN pairs are dropped before correlation. Minimum 3 valid heads required.

    Output schema:
        ["model_name","layer_idx","delta","metric_name",
         "pearson_r","spearman_rho","pearson_std_over_prompts","n_heads","n_prompts"]

    Saves to CORRELATION_PATH via save_results (idempotent merge).
    """
    arch_df = pd.read_parquet(ARCH_METRICS_PATH).reset_index(drop=True)
    mass_df = pd.read_parquet(DIAG_MASS_PATH).reset_index(drop=True)

    arch_df = arch_df[arch_df["model_name"] == model_name]
    mass_df = mass_df[mass_df["model_name"] == model_name]

    # Mean diag_mass per (layer, head, delta) across all prompts
    mean_mass = (
        mass_df
        .groupby(["layer_idx", "head_idx", "delta"])["diag_mass"]
        .mean()
        .reset_index()
    )

    prompts   = mass_df["prompt_id"].unique()
    n_prompts = len(prompts)

    METRIC_COLS = ["subspace_overlap", "trace_norm", "sigma1", "spectral_entropy"]
    layers = sorted(arch_df["layer_idx"].unique())
    results = []

    for layer_idx in tqdm(layers, desc="correlations — layers"):
        arch_layer      = arch_df[arch_df["layer_idx"] == layer_idx]
        mean_mass_layer = mean_mass[mean_mass["layer_idx"] == layer_idx]
        mass_layer      = mass_df[mass_df["layer_idx"] == layer_idx]

        for delta in range(DELTA_MAX + 1):
            arch_d = arch_layer[arch_layer["delta"] == delta].sort_values("head_idx")
            mass_d = mean_mass_layer[mean_mass_layer["delta"] == delta].sort_values("head_idx")

            merged = arch_d.merge(
                mass_d[["head_idx", "diag_mass"]], on="head_idx", how="inner"
            )
            if len(merged) < 3:
                continue

            y_mean  = merged["diag_mass"].values
            n_heads = len(merged)

            for metric_name in METRIC_COLS:
                x = merged[metric_name].values
                valid = ~(np.isnan(x) | np.isnan(y_mean))
                if valid.sum() < 3:
                    continue

                xv, yv = x[valid], y_mean[valid]

                try:
                    pearson_r, _ = stats.pearsonr(xv, yv)
                except Exception:
                    pearson_r = np.nan

                try:
                    spearman_rho, _ = stats.spearmanr(xv, yv)
                except Exception:
                    spearman_rho = np.nan

                # ── Per-prompt Pearson for stability estimate ──────────────
                per_prompt_r = []
                mass_delta_all = mass_layer[mass_layer["delta"] == delta]

                for prompt_id in prompts:
                    mass_p = (
                        mass_delta_all[mass_delta_all["prompt_id"] == prompt_id]
                        .sort_values("head_idx")
                    )
                    merged_p = arch_d.merge(
                        mass_p[["head_idx", "diag_mass"]], on="head_idx", how="inner"
                    )
                    if len(merged_p) < 3:
                        continue

                    xp = merged_p[metric_name].values
                    yp = merged_p["diag_mass"].values
                    vp = ~(np.isnan(xp) | np.isnan(yp))
                    if vp.sum() < 3:
                        continue

                    try:
                        r_p, _ = stats.pearsonr(xp[vp], yp[vp])
                        per_prompt_r.append(r_p)
                    except Exception:
                        pass

                pearson_std = float(np.std(per_prompt_r)) if len(per_prompt_r) > 1 else np.nan

                results.append(dict(
                    model_name=model_name,
                    layer_idx=layer_idx,
                    delta=delta,
                    metric_name=metric_name,
                    pearson_r=float(pearson_r) if not np.isnan(pearson_r) else np.nan,
                    spearman_rho=float(spearman_rho) if not np.isnan(spearman_rho) else np.nan,
                    pearson_std_over_prompts=pearson_std,
                    n_heads=n_heads,
                    n_prompts=n_prompts,
                ))

    return save_results(results, CORRELATION_PATH, CORR_PRIMARY_KEY)


# ── Part 4 ────────────────────────────────────────────────────────────────────

def run(model_name: str = MODEL_NAME, skip_arch: bool = False, skip_mass: bool = False):
    """Top-level orchestrator. Loads model once, calls steps 1 -> 2 -> 3.

    Designed to be called from a Kaggle notebook:
        from arch_diagonal_analysis import run
        run(model_name="Qwen/Qwen3-4B", skip_arch=False, skip_mass=False)

    Args:
        model_name : HuggingFace model identifier (default: config.MODEL_NAME).
        skip_arch  : Skip Part 1 if arch_metrics.parquet already exists.
        skip_mass  : Skip Part 2 if diag_mass.parquet already exists.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = DEVICE_CUDA
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = DEVICE_MPS
    else:
        device = DEVICE_CPU

    print(f"[run] device={device}  model={model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=torch.float16 if device == DEVICE_CUDA else torch.float32,
        device_map="auto",
    )
    model.eval()

    if not skip_arch:
        print("[run] Step 1/3 — architectural metrics (weights only) ...")
        compute_arch_metrics(model, model_name)
    else:
        print("[run] Step 1/3 — skipped (skip_arch=True)")

    if not skip_mass:
        print("[run] Step 2/3 — empirical diagonal mass (forward passes) ...")
        compute_diag_mass(model, tokenizer, model_name)
    else:
        print("[run] Step 2/3 — skipped (skip_mass=True)")

    print("[run] Step 3/3 — correlation analysis ...")
    compute_correlations(model_name)
    print("[run] Done.")