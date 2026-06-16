# =============================================================================
# ATTENTION PATTERN ANALYSIS — Multi-Model, Multi-Dataset
# Models: Mistral-7B-Instruct-v0.3 | Qwen3-4B
# Datasets: Wikitext-103 | FineWeb-Edu
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from collections import defaultdict
from pathlib import Path
import pandas as pd
import matplotlib.colors as mcolors


# =============================================================================
# TARGET FEATURES — all features whose head-dependence we want to quantify
# =============================================================================

TARGET_FEATURES = [
    "diagonal_mass_1",
    "diagonal_mass_5",
    "diagonal_mass_1_shifted_1",
    "diagonal_mass_1_shifted_2",
    "diagonal_mass_1_shifted_3",
    "diagonal_mass_1_shifted_4",
    "sink_mass_token_0",
    "sink_mass_token_1",
    "sink_mass_token_2",
    "sink_mass_token_3",
    "sink_mass_token_4",
    "sink_mass_max",
    "look_back",
    "attention_gini",
    "effective_rank_A",
    "r95_A",
]

# Dataset split labels — "all" means no split
DATASET_SPLITS = ["all", "wikitext", "fineweb"]


# =============================================================================
# GLOBAL PLOT SETTINGS
# =============================================================================

C_WIKI    = "#4878CF"   # blue   — wikitext split
C_FINEWEB = "#E07B39"   # orange — fineweb split

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333",
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.color":        "#CCCCCC",
    "grid.linewidth":    0.5,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.titlesize":    9,
    "axes.titleweight":  "bold",
    "axes.titlepad":     6,
    "font.family":       "DejaVu Sans",
})

N_COLS          = 2
STRIP_THRESHOLD = 80    # kept for _choose_plot_type logic, strip path removed
RANK_PADDING    = 5
SAVE_PLOTS      = True
out_dir         = Path("thesis_images"); out_dir.mkdir(exist_ok=True)

QUARTILE_COLORS = ["#3B4CC0", "#6EC6C6", "#F4A44A", "#D65F5F"]
QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS = {
    "mistral_7b": {
        "label":         "Mistral-7B-Instruct-v0.3",
        "model_name":    "mistralai/Mistral-7B-Instruct-v0.3",          
        "color":         "#2CA02C",              
        "n_heads":       32,
        "n_kv_heads":    8,
        "gqa_ratio":     4,                      
        "head_dim":      128,
        "hidden_size":   4096,
        "num_layers":    32,
        "rope_theta":    1_000_000,
        "ptype_wiki":    "wikitext_wikitext-103-raw-v1_train",
        "ptype_fineweb": "fineweb-edu_sample-10BT_train_stream",
    },
    "qwen3_4b": {
        "label":         "Qwen3-4B",
        "model_name":    "Qwen/Qwen3-4B",             
        "color":         "#9467BD",              
        "n_heads":       32,
        "n_kv_heads":    8,
        "gqa_ratio":     4,                      
        "head_dim":      128,
        "hidden_size":   2560,
        "num_layers":    36,
        "rope_theta":    1_000_000,
        "ptype_wiki":    "wikitext_wikitext-103-raw-v1_train",
        "ptype_fineweb": "fineweb-edu_sample-10BT_train_stream",
    },
}


# =============================================================================
# TAXONOMY
# Sections: model_dependent | input_dependent | target
# Each group defines features, axis bounds, and display options.
# rank_max and hidden_size are resolved at runtime from model_cfg.
# =============================================================================

TAXONOMY = {
    "model_dependent": {
        "Weight Matrix Ranks Wq": {
            "features":    ["effective_rank_Wq", "r95_Wq"],
            "bounds":      {},
            "rank_max":    "head_dim",          
            "use_logy":    False,
            "use_density": True,
            "show_split":  False,
            "notes": "Per-head slice of W_q.",
        },
        "Weight Matrix Ranks Wk Wv": {
            "features":    ["effective_rank_Wk", "r95_Wk", "effective_rank_Wv", "r95_Wv"],
            "bounds":      {},
            "rank_max":    "head_dim",
            "use_logy":    False,
            "use_density": True,
            "show_split":  False,
            "notes": "Per-KV-head slice of W_k/W_v (GQA).",
        },
        "Gini Spectral Concentration": {
            "features":    ["gini_left_Wq", "gini_right_Wq", "gini_left_Wk", "gini_right_Wk"],
            "bounds":      {f: (0.0, 1.0) for f in
                            ["gini_left_Wq", "gini_right_Wq", "gini_left_Wk", "gini_right_Wk"]},
            "use_logy":    False,
            "use_density": True,
            "show_split":  False,
            "notes": "Gini over left/right singular vectors of W_q/W_k. In [0,1].",
        },
        "RMSNorm": {
            "features":    ["rmsnorm_gamma_norm"],
            "bounds":      {"rmsnorm_gamma_norm": (0.0, None)},
            "use_logy":    False,
            "use_density": True,
            "show_split":  False,
            "notes": "L2 norm of RMSNorm gamma. Fixed per layer.",
        },
        "RoPE Structure": {
            "features":    ["rope_pair_var_Wq",       "rope_pair_var_Wk",
                            "rope_pair_max_ratio_Wq", "rope_pair_max_ratio_Wk",
                            "rope_freq_com_Wq",       "rope_freq_com_Wk"],
            "bounds":      {
                "rope_pair_var_Wq":       (0, None),
                "rope_pair_var_Wk":       (0, None),
                "rope_pair_max_ratio_Wq": (0, None),
                "rope_pair_max_ratio_Wk": (0, None),
                "rope_freq_com_Wq":       (0, 63.0),
                "rope_freq_com_Wk":       (0, 63.0),
            },
            "use_logy":    False,
            "use_density": True,
            "show_split":  False,
            "notes": "rope_freq_com in [0, head_dim/2 - 1]. rope_pair_max_ratio: top RoPE pair dominance.",
        },
        "RoPE-aware QK Alignment": {
            "features":    ["compute_WqRWk_alignment_delta_0"],
            "bounds":      {"compute_WqRWk_alignment_delta_0": (0, 1)},
            "use_logy":    False,
            "use_density": True,
            "show_split":  False,
            "notes": "Cosine alignment W_q * R(delta=0) * W_k^T. Pure weight property.",
        },
    },

    "input_dependent": {
        "Hidden State Rank": {
            "features":    ["effective_rank_H", "r95_H"],
            "bounds":      {},
            "rank_max":    "hidden_size",        # resolved from model_cfg["hidden_size"]
            "use_logy":    False,
            "use_density": True,
            "show_split":  True,
            "notes": "H shape (seq_len, hidden_size). rank_max=min(seq_len, hidden_size).",
        },
        "Projected Q/K Ranks": {
            "features":    ["effective_rank_Q", "r95_Q", "effective_rank_K", "r95_K"],
            "bounds":      {},
            "rank_max":    "head_dim",
            "use_logy":    False,
            "use_density": True,
            "show_split":  True,
            "notes": "Q=H*W_q^T, K=H*W_k^T per head. Shape (seq, head_dim).",
        },
        "Temporal Similarity": {
            "features":    ["q_sim_consecutive", "k_sim_consecutive"],
            "bounds":      {"q_sim_consecutive": (0, 1), "k_sim_consecutive": (0, 1)},
            "use_logy":    False,
            "use_density": True,
            "show_split":  True,
            "notes": "Mean cosine sim between consecutive token rows of Q/K.",
        },
        "SVD Alignment H vs W": {
            "features":    ["svd_alignment_H_Wq", "svd_alignment_H_Wk"],
            "bounds":      {"svd_alignment_H_Wq": (0, 1), "svd_alignment_H_Wk": (0, 1)},
            "use_logy":    False,
            "use_density": True,
            "show_split":  True,
            "notes": "Cosine sim of top singular vector of H vs W_q/W_k.",
        },
    },

    "target": {
        "Attention Map Diagonal Mass": {
            "features":    ["diagonal_mass_1",       "diagonal_mass_5",
                            "diagonal_mass_1_shifted_1", "diagonal_mass_1_shifted_2",
                            "diagonal_mass_1_shifted_3", "diagonal_mass_1_shifted_4"],
            "bounds":      {f: (0, 1) for f in
                            ["diagonal_mass_1", "diagonal_mass_5",
                             "diagonal_mass_1_shifted_1", "diagonal_mass_1_shifted_2",
                             "diagonal_mass_1_shifted_3", "diagonal_mass_1_shifted_4"]},
            "use_logy":    True,
            "use_density": True,
            "show_split":  True,
            "notes": "Attention mass on main diagonal (width 1 or 5) and sub-diagonals.",
        },
        "Attention Map Sink Mass": {
            "features":    ["sink_mass_token_0", "sink_mass_token_1", "sink_mass_token_2",
                            "sink_mass_token_3", "sink_mass_token_4", "sink_mass_max"],
            "bounds":      {f: (0, 1) for f in
                            ["sink_mass_token_0", "sink_mass_token_1", "sink_mass_token_2",
                             "sink_mass_token_3", "sink_mass_token_4", "sink_mass_max"]},
            "use_logy":    True,
            "use_density": True,
            "show_split":  True,
            "notes": "Attention mass on first 5 tokens (sinks). sink_mass_max = row-wise max.",
        },
        "Attention Map Structure": {
            "features":    ["look_back", "attention_gini"],
            "bounds":      {"look_back": (0, 1), "attention_gini": (0, 1)},
            "use_logy":    False,
            "use_density": True,
            "show_split":  True,
            "notes": "look_back: mean backward distance norm. attention_gini: row Gini.",
        },
        "Attention Matrix Rank": {
            "features":    ["effective_rank_A", "r95_A"],
            "bounds":      {},
            "rank_max":    512,                  # static: post-softmax A, seq-level rank cap
            "use_logy":    False,
            "use_density": True,
            "show_split":  True,
            "notes": "Rank of post-softmax A. Low=sparse, High=diffuse.",
        },
    },
}


# =============================================================================
# FEATURE DEDUPLICATION MAP
# Specifies which columns uniquely identify a row for weight-derived features.
# "kv_head" is derived at runtime using model_cfg["gqa_ratio"].
# =============================================================================

FEATURE_DEDUP = {
    # Q-head features — unique per (layer, head)
    "effective_rank_Wq":               ["layer_idx", "head_idx"],
    "r95_Wq":                          ["layer_idx", "head_idx"],
    "gini_left_Wq":                    ["layer_idx", "head_idx"],
    "gini_right_Wq":                   ["layer_idx", "head_idx"],
    "rope_pair_var_Wq":                ["layer_idx", "head_idx"],
    "rope_pair_max_ratio_Wq":          ["layer_idx", "head_idx"],
    "rope_freq_com_Wq":                ["layer_idx", "head_idx"],
    "compute_WqRWk_alignment_delta_0": ["layer_idx", "head_idx"],
    # KV-head features — unique per (layer, kv_head)
    "effective_rank_Wk":               ["layer_idx", "kv_head"],
    "r95_Wk":                          ["layer_idx", "kv_head"],
    "effective_rank_Wv":               ["layer_idx", "kv_head"],
    "r95_Wv":                          ["layer_idx", "kv_head"],
    "gini_left_Wk":                    ["layer_idx", "kv_head"],
    "gini_right_Wk":                   ["layer_idx", "kv_head"],
    "rope_pair_var_Wk":                ["layer_idx", "kv_head"],
    "rope_pair_max_ratio_Wk":          ["layer_idx", "kv_head"],
    "rope_freq_com_Wk":                ["layer_idx", "kv_head"],
    # Layer-level features — unique per layer
    "rmsnorm_gamma_norm":              ["layer_idx"],
}


# =============================================================================
# HELPER: resolve rank_max from model_cfg
# =============================================================================

def _resolve_rank_max(meta, model_cfg):
    """
    Resolves rank_max to an integer.
    If rank_max is a string key (e.g. "head_dim"), look it up in model_cfg.
    If it is already an int (or absent), return it directly.
    """
    rm = meta.get("rank_max", None)
    if isinstance(rm, str):
        return model_cfg[rm]
    return rm


# =============================================================================
# HELPER: prepare values for a single feature
# Returns a dict: {"wikitext": array, "fineweb": array} or {"all": array}
# =============================================================================

def _prepare_vals(df, feat, model_cfg, show_split=False):
    dedup_cols = FEATURE_DEDUP.get(feat, None)

    if dedup_cols is not None:
        # Derive kv_head on the fly if needed and not yet present
        if "kv_head" in dedup_cols and "kv_head" not in df.columns:
            df = df.copy()
            df["kv_head"] = df["head_idx"] // model_cfg["gqa_ratio"]
        subset = [c for c in dedup_cols if c in df.columns]
        sub = df.drop_duplicates(subset=subset) if subset else df
    else:
        sub = df

    if show_split:
        return {
            "wikitext": sub[sub["prompt_source"] == model_cfg["ptype_wiki"]][feat].dropna().values,
            "fineweb":  sub[sub["prompt_source"] == model_cfg["ptype_fineweb"]][feat].dropna().values,
        }
    else:
        return {"all": sub[feat].dropna().values}


# =============================================================================
# HELPER: group rank features by matrix suffix (e.g. "Wq", "H")
# =============================================================================

def _group_rank_features_by_suffix(features):
    groups = defaultdict(list)
    for f in features:
        matched = False
        for prefix in ("effective_rank_", "r95_"):
            if f.startswith(prefix):
                groups[f[len(prefix):]].append(f)
                matched = True
                break
        if not matched:
            groups[f].append(f)
    return dict(groups)


# =============================================================================
# HELPER: compute shared axis bounds for rank features
# =============================================================================

def _compute_rank_bounds(vals_list, rank_max, padding=RANK_PADDING):
    all_v = np.concatenate([v for v in vals_list if len(v)])
    if len(all_v) == 0:
        return 1, (rank_max if rank_max is not None else 1)
    lo = max(1, int(np.floor(all_v.min())) - padding)
    hi = int(np.ceil(all_v.max())) + padding
    if rank_max is not None:
        hi = min(hi, rank_max)
    return lo, hi


# =============================================================================
# HELPER: build per-feature axis bounds dict
# =============================================================================

def _build_feat_bounds(valid, meta, df, model_cfg):
    bounds   = meta["bounds"]
    rank_max = _resolve_rank_max(meta, model_cfg)
    feat_bounds = {}

    if rank_max is not None:
        suffix_groups = _group_rank_features_by_suffix(valid)
        for suffix, suffix_feats in suffix_groups.items():
            group_vals = []
            for f in suffix_feats:
                sv = _prepare_vals(df, f, model_cfg)
                group_vals.extend(v for v in sv.values() if len(v))
            lo_p, hi_p = _compute_rank_bounds(group_vals, rank_max)
            for f in suffix_feats:
                feat_bounds[f] = (lo_p, hi_p)
    else:
        for f in valid:
            sv    = _prepare_vals(df, f, model_cfg)
            all_v = np.concatenate(list(sv.values()))
            if len(all_v) == 0:
                feat_bounds[f] = (0.0, 1.0)
                continue
            lo, hi = bounds.get(f, (None, None))
            lo_p   = lo if lo is not None else float(all_v.min()) - abs(float(all_v.min())) * 0.02
            hi_p   = hi if hi is not None else float(all_v.max()) * 1.02
            feat_bounds[f] = (lo_p, hi_p)

    return feat_bounds


# =============================================================================
# HELPER: automatic bin computation (Freedman-Diaconis / Scott / Sturges)
# =============================================================================

def _compute_bins(vals_list, lo_p, hi_p, min_bins=10, max_bins=150):
    all_v = np.concatenate([np.clip(v, lo_p, hi_p) for v in vals_list if len(v)])
    n     = len(all_v)
    r     = hi_p - lo_p

    if n < 4 or r == 0:
        return np.linspace(lo_p, hi_p, min_bins + 1)

    iqr = np.percentile(all_v, 75) - np.percentile(all_v, 25)
    std = np.std(all_v)

    n_fd = int(np.ceil(r / (2.0 * iqr * n**(-1/3)))) if iqr > 0 else max_bins
    n_sc = int(np.ceil(r / (3.49 * std * n**(-1/3)))) if std > 0 else max_bins
    n_st = int(np.ceil(np.log2(n) + 1))

    if iqr > 0 and (r / iqr) > 50:
        n_bins = max(n_sc, n_st)
    elif iqr == 0:
        n_bins = max(int(np.sqrt(n)), n_st)
    else:
        n_bins = int(np.median([n_fd, n_sc, n_st]))

    return np.linspace(lo_p, hi_p, int(np.clip(n_bins, min_bins, max_bins)) + 1)


# =============================================================================
# SAVE HELPER
# =============================================================================

def _save_fig(fig, name, model_cfg):
    model_dir = out_dir / model_cfg["model_name"]
    model_dir.mkdir(exist_ok=True)
    safe = (name.lower()
            .replace(" ", "_").replace("(", "").replace(")", "")
            .replace(",", "").replace("/", "_").replace("—", "")
            .replace("&", "").replace("__", "_").strip("_"))
    fig.savefig(model_dir / f"{safe}.png", dpi=150, bbox_inches="tight")


# =============================================================================
# CORE PLOT: one figure per feature
# =============================================================================

def _plot_feature(feat, feat_bounds, meta, df, model_cfg):
    """
    Renders a single histogram for one feature.
    - model_dependent features: layer-quartile stacked histograms (no dataset split).
    - input_dependent / target features: overlaid histograms split by dataset.
    Returns the matplotlib Figure object.
    """
    from IPython.display import display

    palette = {"wikitext": C_WIKI, "fineweb": C_FINEWEB, "all": model_cfg["color"]}
    labels  = {"wikitext": "Wikitext-103", "fineweb": "FineWeb-Edu",
               "all": f"all heads ({model_cfg['label']})"}

    use_logy    = meta["use_logy"]
    use_density = meta["use_density"]
    lo_p, hi_p  = feat_bounds[feat]

    use_layer_quartiles = (
        not meta["show_split"]
        and "layer_idx" in df.columns
        and df["layer_idx"].nunique() >= 4
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)

    # ── Branch A: layer-quartile stacked histogram (model_dependent) ─────────
    if use_layer_quartiles:
        layers     = sorted(df["layer_idx"].unique())
        q_size     = len(layers) // 4
        layer_to_q = {l: min(i // q_size, 3) for i, l in enumerate(layers)}

        dedup = FEATURE_DEDUP.get(feat, None)
        tmp   = df.copy()
        if dedup is not None:
            if "kv_head" in dedup and "kv_head" not in tmp.columns:
                tmp["kv_head"] = tmp["head_idx"] // model_cfg["gqa_ratio"]
            key_cols = [c for c in dedup if c in tmp.columns]
            tmp = tmp.drop_duplicates(subset=key_cols)

        tmp = tmp[["layer_idx", feat]].dropna()
        if tmp.empty:
            plt.close(fig)
            return None

        tmp["quartile"] = tmp["layer_idx"].map(layer_to_q)
        q_vals = [
            np.clip(tmp.loc[tmp["quartile"] == q, feat].values, lo_p, hi_p)
            for q in range(4)
        ]
        q_vals = [v for v in q_vals if len(v) > 0]
        if not q_vals:
            plt.close(fig)
            return None

        bins = _compute_bins(q_vals, lo_p, hi_p)
        ax.hist(q_vals, bins=bins, stacked=True,
                color=QUARTILE_COLORS[:len(q_vals)],
                label=QUARTILE_LABELS[:len(q_vals)],
                alpha=0.85, density=use_density, edgecolor="none")
        loc_str = "upper left" if ("rank" in feat.lower() or "q_sim" in feat.lower() or "k_sim" in feat.lower() or "r95" in feat.lower() or "gini" in feat.lower() or 'look' in feat.lower()) else "upper right"
        ax.legend(title="Layer depth", title_fontsize=7, loc=loc_str, fontsize=7)
        split_tag = "[layer quartiles]"

    # ── Branch B: per-dataset split histogram (input_dependent / target) ─────
    else:
        split_vals = _prepare_vals(df, feat, model_cfg, meta["show_split"])
        all_v      = np.concatenate(list(split_vals.values()))

        if len(all_v) == 0:
            plt.close(fig)
            return None

        bins = _compute_bins(list(split_vals.values()), lo_p, hi_p)

        for key, vals in split_vals.items():
            vc = np.clip(vals, lo_p, hi_p)
            ax.hist(vc, bins=bins, color=palette[key], alpha=0.65,
                    label=labels[key], density=use_density)
            if len(vc):
                ax.axvline(np.mean(vc),   color=palette[key], lw=1.4, ls="-")
                ax.axvline(np.median(vc), color=palette[key], lw=1.4, ls="--")

        loc_str = "upper left" if ("rank" in feat.lower() or "q_sim" in feat.lower() or "k_sim" in feat.lower() or "r95" in feat.lower() or "gini" in feat.lower() or 'look' in feat.lower()) else "upper right"
        ax.legend(title="— mean   -- median",
                  title_fontsize=7, loc=loc_str, fontsize=7)
        split_tag = "[wiki vs fineweb]" if meta["show_split"] else "[model only]"

    ax.set_xlim(lo_p, hi_p)
    if use_logy:
        ax.set_yscale("log")
    ax.set_ylabel("density" if use_density else "count")
    ax.set_xlabel(feat.replace("_", " "))

    logy_tag   = "  [log y]"   if use_logy    else ""
    density_tag= "  [density]" if use_density else "  [count]"
    ax.set_title(
        f"{feat.replace('_', ' ')}{logy_tag}{density_tag}  {split_tag}\n"
        f"{model_cfg['label']}",
        pad=5,
    )

    return fig



# =============================================================================
# GROUP ENTRY POINT — one figure per feature, one figure set per model
# =============================================================================

def plot_group(group_name, meta, df, model_cfg):
    """
    Iterates over every valid feature in the group and produces one
    independent figure per feature. Figures are displayed inline and
    optionally saved under out_dir/<model_name>/.
    """
    from IPython.display import display

    features = meta["features"]
    valid    = [f for f in features if f in df.columns]
    missing  = [f for f in features if f not in df.columns]

    if missing:
        print(f"  [SKIP] {group_name} — missing columns: {missing}")
    if not valid:
        return

    feat_bounds = _build_feat_bounds(valid, meta, df, model_cfg)

    for feat in valid:
        fig = _plot_feature(feat, feat_bounds, meta, df, model_cfg)
        if fig is None:
            print(f"  [EMPTY] {feat} — no data after filtering.")
            continue
        if SAVE_PLOTS:
            _save_fig(fig, f"{group_name}_{feat}", model_cfg)
        display(fig)
        plt.close(fig)


# =============================================================================
# MAIN ANALYSIS LOOP
# Iterates over M x TAXONOMY, where M is the set of active model keys.
# =============================================================================

def run_analysis(df_combined, model_keys=None):
    """
    Parameters
    ----------
    df_combined : pd.DataFrame
        Combined dataframe with a "model_name" column that matches
        MODEL_CONFIGS keys (e.g. "mistral_7b", "qwen3_4b").
    model_keys : list[str] | None
        Subset of MODEL_CONFIGS keys to analyse. Defaults to all.
    """
    if model_keys is None:
        model_keys = list(MODEL_CONFIGS.keys())

    for model_key in model_keys:
        cfg = MODEL_CONFIGS[model_key]
        df  = df_combined[df_combined["model_name"] == cfg["model_name"]].copy()

        if df.empty:
            print(f"[WARNING] No data found for model '{model_key}'. Skipping.")
            continue

        print(f"\n{'═' * 64}")
        print(f"  Model : {cfg['label']}")
        print(f"  Rows  : {len(df):,}  |  Layers: {df['layer_idx'].nunique()}  "
              f"|  Heads: {df['head_idx'].nunique()}")
        print(f"{'═' * 64}")

        for section, groups in TAXONOMY.items():
            print(f"\n── {section.upper()} ──────────────────────────────────")
            for group_name, meta in groups.items():
                print(f"  › {group_name}")
                plot_group(group_name, meta, df, cfg)



# =============================================================================
# PER-HEAD BOXPLOT — Wikitext vs FineWeb
# One box per dataset per cell. Layout adapts to layer/head selection.
# =============================================================================

def plot_target_boxplot_per_head(
    df,
    model_cfg:          dict,
    feat:               str = "attention_gini",
    layer:              "int | list | None" = None,
    head:               "int | None" = None,
    figsize_per_cell:   tuple = (2.4, 2.0),
    save:               bool = False,
):
    """
    Boxplot of a chosen feature (e.g. attention_gini, diagonal_mass_1)
    for each attention head, split by dataset (Wikitext vs FineWeb-Edu).

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe for a SINGLE model (already filtered by model_name,
        or the full combined df — model filtering is NOT done here).
    model_cfg : dict
        Entry from MODEL_CONFIGS (e.g. MODEL_CONFIGS["mistral_7b"]).
    feat : str
        Column name of the feature to plot.
    layer : int | list[int] | None
        - None  → all layers (multi-layer layout: n_layers × n_heads).
        - int   → single layer (single-layer layout: 4 cols × ceil(n_heads/4) rows).
        - list  → specified layers (multi-layer layout).
    head : int | None
        - None  → all heads (default).
        - int   → single head; each layer becomes one row with a single cell.
    figsize_per_cell : tuple
        (width, height) in inches for each (layer, head) cell.
    save : bool
        If True, saves to out_dir/<model_name>/.
    """
    from IPython.display import display

    if feat not in df.columns:
        print(f"[ERROR] Feature '{feat}' not found in DataFrame.")
        return

    ptype_wiki    = model_cfg["ptype_wiki"]
    ptype_fineweb = model_cfg["ptype_fineweb"]

    df_wiki = df[df["prompt_source"] == ptype_wiki]
    df_fine = df[df["prompt_source"] == ptype_fineweb]

    # ── Layer selection ───────────────────────────────────────────────────────
    all_layers = sorted(df["layer_idx"].unique())

    if layer is None:
        layers = all_layers
    elif isinstance(layer, int):
        if layer not in all_layers:
            print(f"[ERROR] Layer {layer} not found. Available: {all_layers}")
            return
        layers = [layer]
    elif isinstance(layer, list):
        layers = [l for l in layer if l in all_layers]
        if not layers:
            print("[ERROR] None of the specified layers found in DataFrame.")
            return
    else:
        raise ValueError("'layer' must be None, int, or list[int].")

    # ── Head selection ────────────────────────────────────────────────────────
    all_heads = sorted(df["head_idx"].unique())

    if head is None:
        heads = all_heads
    elif isinstance(head, int):
        if head not in all_heads:
            print(f"[ERROR] Head {head} not found. Available: {all_heads}")
            return
        heads = [head]
    else:
        raise ValueError("'head' must be None or int.")

    n_h = len(heads)
    n_l = len(layers)

    # ── Layout selection ──────────────────────────────────────────────────────
    single_layer_mode = (n_l == 1)
    single_head_mode  = (n_h == 1)

    if single_head_mode:
        # One cell per layer, stacked vertically in a single column
        n_cols = 1
        n_rows = n_l
    elif single_layer_mode:
        # All heads of one layer: 4-column grid
        n_cols = 4
        n_rows = int(np.ceil(n_h / n_cols))
    else:
        # Full grid: rows = layers, cols = heads
        n_cols = n_h
        n_rows = n_l

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        constrained_layout=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    # ── Global median range for colormap normalisation ────────────────────────
    medians_global = (
        df[df["layer_idx"].isin(layers) & df["head_idx"].isin(heads)]
        .groupby(["layer_idx", "head_idx"])[feat]
        .median()
        .dropna()
    )
    vmin = medians_global.min() if len(medians_global) else 0.0
    vmax = medians_global.max() if len(medians_global) else 1.0
    if vmin == vmax:
        vmin, vmax = vmin - 0.01, vmax + 0.01

    cmap = plt.get_cmap("RdYlGn")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rng  = np.random.default_rng(42)

    # ── Cell drawing helper ───────────────────────────────────────────────────
    def _draw_cell(ax, layer_idx, head_idx, title_str, ylabel_str=None):
        sub_w = df_wiki[
            (df_wiki["layer_idx"] == layer_idx) & (df_wiki["head_idx"] == head_idx)
        ][feat].dropna()
        sub_f = df_fine[
            (df_fine["layer_idx"] == layer_idx) & (df_fine["head_idx"] == head_idx)
        ][feat].dropna()

        if sub_w.empty and sub_f.empty:
            ax.set_visible(False)
            return

        for pos, sub, c_line in [(1, sub_w, C_WIKI), (2, sub_f, C_FINEWEB)]:
            if sub.empty:
                continue
            median_val = sub.median()
            face_color = cmap(norm(median_val))

            ax.boxplot(
                sub.values, positions=[pos], widths=0.45,
                patch_artist=True, notch=False, showfliers=False,
                medianprops=dict(color="#222", lw=1.8),
                whiskerprops=dict(color=c_line, lw=0.8),
                capprops=dict(color=c_line, lw=0.8),
                boxprops=dict(facecolor=face_color, alpha=0.80,
                              linewidth=0.8, edgecolor=c_line),
            )
            jitter = rng.uniform(-0.12, 0.12, size=len(sub))
            ax.scatter(
                np.full(len(sub), pos) + jitter, sub.values,
                s=6, alpha=0.55, color=c_line, edgecolors="none", zorder=3,
            )
            fmt = (f"{median_val:.1e}"
                   if 0 < abs(median_val) < 0.01
                   else f"{median_val:.2f}")
            ax.text(pos + 0.30, median_val, fmt,
                    va="center", ha="left", fontsize=5.5,
                    color="#333", zorder=4)

        title_fs = 8 if (single_layer_mode or single_head_mode) else 6
        ax.set_title(title_str, fontsize=title_fs, pad=3)
        ax.set_xlim(0.4, 2.9)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["W", "F"], fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_ylim(vmin - (vmax - vmin) * 0.05,
                    vmax + (vmax - vmin) * 0.10)
        ax.grid(True, axis="y", lw=0.3, alpha=0.4)
        if ylabel_str:
            ax.set_ylabel(ylabel_str, fontsize=6, rotation=0,
                          labelpad=20, va="center")

    # ── Populate cells ────────────────────────────────────────────────────────
    if single_head_mode:
        # One row per layer, single head
        h = heads[0]
        for row, layer_idx in enumerate(layers):
            ylabel = f"L{layer_idx}"
            title  = f"Head {h}" if row == 0 else ""
            _draw_cell(axes[row, 0], layer_idx, h,
                       title_str=title, ylabel_str=ylabel)

    elif single_layer_mode:
        layer_idx = layers[0]
        for idx, h in enumerate(heads):
            row, col = divmod(idx, n_cols)
            _draw_cell(axes[row, col], layer_idx, h,
                       title_str=f"Head {h}")
        # Hide unused cells
        for idx in range(n_h, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

    else:
        # Full grid
        for row, layer_idx in enumerate(layers):
            for col, h in enumerate(heads):
                ylabel = f"L{layer_idx}" if col == 0 else None
                title  = f"H{h}"         if row == 0 else ""
                _draw_cell(axes[row, col], layer_idx, h,
                           title_str=title, ylabel_str=ylabel)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.4, pad=0.005, aspect=40)
    cbar.set_label(f"Median {feat}", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Dataset legend ────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=C_WIKI,    lw=2, label="Wikitext-103"),
        Line2D([0], [0], color=C_FINEWEB, lw=2, label="FineWeb-Edu"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.45, -0.02))

    # ── Title ─────────────────────────────────────────────────────────────────
    if single_head_mode:
        scope_tag = (f"Head {heads[0]} — "
                     + (f"Layer {layers[0]}" if n_l == 1
                        else f"Layers {layers[0]}–{layers[-1]}"))
    elif single_layer_mode:
        scope_tag = f"All Heads — Layer {layers[0]}"
    else:
        scope_tag = (f"All Heads — "
                     + (f"Layer {layers[0]}" if n_l == 1
                        else f"Layers {layers[0]}–{layers[-1]}"))

    fig.suptitle(
        f"{feat} per Head — Wikitext vs FineWeb\n"
        f"{model_cfg['label']}  |  {scope_tag}",
        fontsize=10, fontweight="bold",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        safe_feat = feat.replace(" ", "_")
        layer_str = (f"L{layers[0]}" if n_l == 1
                     else f"L{layers[0]}-{layers[-1]}")
        head_str  = f"_H{heads[0]}" if single_head_mode else ""
        fname     = f"{safe_feat}_boxplot{head_str}_{layer_str}"
        _save_fig(fig, fname, model_cfg)

    display(fig)
    plt.close(fig)



# =============================================================================
# CORE: ICC(1,1) estimator
# =============================================================================

def _icc_one_way(groups: list[np.ndarray]) -> dict:
    """
    Computes one-way random effects ICC(1,1) from a list of group arrays.
    Returns a dict with ICC and the underlying MS terms.

    Parameters
    ----------
    groups : list of 1-D np.ndarray
        One array per head, containing all observations for that head.

    Returns
    -------
    dict with keys: icc, ms_between, ms_within, n_harmonic, k, n
    """
    k = len(groups)
    n = sum(len(g) for g in groups)

    if k < 2 or n <= k:
        return {"icc": np.nan, "ms_between": np.nan, "ms_within": np.nan,
                "n_harmonic": np.nan, "k": k, "n": n}

    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within  = sum(((g - g.mean()) ** 2).sum() for g in groups)

    ms_between = ss_between / (k - 1)
    ms_within  = ss_within  / (n - k)

    # Harmonic mean of group sizes (corrects for unbalanced design)
    n_harmonic = (n - sum(len(g) ** 2 / n for g in groups)) / (k - 1)

    icc = (ms_between - ms_within) / (ms_between + (n_harmonic - 1) * ms_within)
    icc = float(np.clip(icc, 0.0, 1.0))   # floor at 0 (negative ICC = no structure)

    return {
        "icc":         round(icc, 6),
        "ms_between":  round(ms_between, 6),
        "ms_within":   round(ms_within, 6),
        "n_harmonic":  round(n_harmonic, 2),
        "k":           k,
        "n":           n,
    }


# =============================================================================
# SINGLE-CALL WRAPPER: one feature × one model subset × one dataset split
# =============================================================================

def compute_icc(df: pd.DataFrame,
                feat: str,
                model_cfg: dict,
                dataset: str = "all") -> dict:
    """
    Computes ICC(1,1) for `feat` grouped by head_idx.

    Parameters
    ----------
    df         : DataFrame for a single model (already filtered by model_name).
    feat       : Column name of the target feature.
    model_cfg  : Entry from MODEL_CONFIGS.
    dataset    : "all" | "wikitext" | "fineweb"

    Returns
    -------
    dict with all ICC stats plus metadata.
    """
    if feat not in df.columns:
        return None

    if dataset == "wikitext":
        sub = df[df["prompt_source"] == model_cfg["ptype_wiki"]]
    elif dataset == "fineweb":
        sub = df[df["prompt_source"] == model_cfg["ptype_fineweb"]]
    else:
        sub = df

    sub = sub[["head_idx", feat]].dropna()
    if sub.empty:
        return None

    groups = [grp[feat].values for _, grp in sub.groupby("head_idx")]
    result = _icc_one_way(groups)

    return {
        "model":   model_cfg["label"],
        "feature": feat,
        "dataset": dataset,
        **result,
    }


# =============================================================================
# SYSTEMATIC RUN: all target features × all models × all dataset splits
# =============================================================================
def run_icc_analysis_per_layer(df_combined, model_keys=None,
                                features=None, splits=None):
    """
    Computes ICC(1,1) grouped by head_idx, stratified per layer.
    Returns one row per (model, layer, feature, dataset).
    """
    if model_keys is None: model_keys = list(MODEL_CONFIGS.keys())
    if features   is None: features   = TARGET_FEATURES
    if splits     is None: splits     = DATASET_SPLITS

    records = []
    for model_key in model_keys:
        cfg      = MODEL_CONFIGS[model_key]
        df_model = df_combined[df_combined["model_name"] == cfg["model_name"]].copy()
        if df_model.empty:
            continue

        layers = sorted(df_model["layer_idx"].unique())
        print(f"\n── {cfg['label']} — {len(layers)} layers ──")

        for layer in layers:
            df_layer = df_model[df_model["layer_idx"] == layer]

            for feat in features:
                if feat not in df_layer.columns:
                    continue
                for ds in splits:
                    if ds == "wikitext":
                        sub = df_layer[df_layer["prompt_source"] == cfg["ptype_wiki"]]
                    elif ds == "fineweb":
                        sub = df_layer[df_layer["prompt_source"] == cfg["ptype_fineweb"]]
                    else:
                        sub = df_layer

                    sub    = sub[["head_idx", feat]].dropna()
                    groups = [g[feat].values for _, g in sub.groupby("head_idx")]
                    result = _icc_one_way(groups)

                    records.append({
                        "model":   cfg["label"],
                        "layer":   layer,
                        "feature": feat,
                        "dataset": ds,
                        **result,
                    })

        # Quick diagnostic: mean ICC across layers per feature
        df_tmp = pd.DataFrame([r for r in records
                                if r["model"] == cfg["label"]
                                and r["dataset"] == "all"])
        if not df_tmp.empty:
            summary = df_tmp.groupby("feature")["icc"].agg(["mean","max","min"])
            print(summary.sort_values("mean", ascending=False).round(4).to_string())

    return pd.DataFrame(records)

# =============================================================================
# VISUALIZATION: heatmap  —  rows=features, cols=models×splits
# =============================================================================
def plot_icc_per_layer(df_icc_layer, features=None, save=False):
    """
    Line plot: x=layer_idx, y=ICC, one line per dataset split.
    One subplot per feature.
    """
    from IPython.display import display

    if features is None:
        features = df_icc_layer["feature"].unique().tolist()

    models  = df_icc_layer["model"].unique()
    n_feat  = len(features)
    n_cols  = min(4, n_feat)
    n_rows  = int(np.ceil(n_feat / n_cols))

    for model_label in models:
        df_m = df_icc_layer[df_icc_layer["model"] == model_label]

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(5 * n_cols, 3 * n_rows),
                                  constrained_layout=True)
        axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

        split_styles = {
            "all":      ("#555555", "-",  1.8),
            "wikitext": (C_WIKI,    "--", 1.2),
            "fineweb":  (C_FINEWEB, "--", 1.2),
        }

        for idx, feat in enumerate(features):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            df_f = df_m[df_m["feature"] == feat]

            for ds, (color, ls, lw) in split_styles.items():
                df_ds = df_f[df_f["dataset"] == ds].sort_values("layer")
                if df_ds.empty: continue
                ax.plot(df_ds["layer"], df_ds["icc"],
                        color=color, ls=ls, lw=lw, label=ds)

            # Reference thresholds
            for thresh, alpha in [(0.50, 0.25), (0.75, 0.20), (0.90, 0.15)]:
                ax.axhline(thresh, color="#999", lw=0.6, ls=":", alpha=0.8)

            ax.set_ylim(-0.02, 1.02)
            ax.set_title(feat.replace("_", " "), fontsize=8)
            ax.set_xlabel("layer", fontsize=7)
            ax.set_ylabel("ICC(1,1)", fontsize=7) if col == 0 else None
            ax.tick_params(labelsize=6)

        # Hide unused axes
        for idx in range(len(features), n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r, c].set_visible(False)

        handles = [plt.Line2D([0],[0], color=c, ls=ls, lw=lw, label=ds)
                   for ds, (c, ls, lw) in split_styles.items()]
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(
            f"ICC(1,1) per Layer — {model_label}\n"
            f"Head identity variance explained, stratified by layer",
            fontsize=10, fontweight="bold"
        )

        if save:
            safe = model_label.lower().replace("-","_").replace(".","")
            out_dir.mkdir(exist_ok=True)
            fig.savefig(out_dir / f"icc_per_layer_{safe}.png",
                        dpi=150, bbox_inches="tight")
        display(fig)
        plt.close(fig)

# =============================================================================
# SUMMARY TABLE: formatted for thesis / notebook reporting
# =============================================================================

def icc_summary_table_per_layer(df_icc: pd.DataFrame,
                                 agg: str = "mean") -> pd.DataFrame:
    """
    Wide-format summary table of per-layer ICC results.

    Aggregates ICC values across layers using `agg` ("mean", "max", or "min"),
    then pivots to:
        rows    = features  (sorted by descending aggregated ICC)
        columns = model × dataset  (ICC value + qualitative rating for "all" split)

    Rating thresholds — Koo & Mae (2016):
        < 0.50          → "poor"
        0.50 – 0.75     → "moderate"
        0.75 – 0.90     → "good"
        ≥ 0.90          → "excellent"

    Parameters
    ----------
    df_icc : pd.DataFrame
        Output of run_icc_analysis_per_layer().
        Must contain columns: model, layer, feature, dataset, icc.
    agg : str
        Aggregation function applied across layers before pivoting.
        One of "mean" (default), "max", "min".

    Returns
    -------
    pd.DataFrame  — wide format, ready for .to_latex() or .to_csv().
    """
    assert agg in ("mean", "max", "min"), "agg must be 'mean', 'max', or 'min'."

    def _rate(v: float) -> str:
        if pd.isna(v):  return "—"
        if v < 0.50:    return "poor"
        if v < 0.75:    return "moderate"
        if v < 0.90:    return "good"
        return "excellent"

    # ── Step 1: aggregate over layers ────────────────────────────────────────
    df_agg = (
        df_icc
        .groupby(["model", "feature", "dataset"])["icc"]
        .agg(agg)
        .reset_index()
        .rename(columns={"icc": f"icc_{agg}"})
    )

    # ── Step 2: build column key  "ModelLabel | dataset (agg)" ───────────────
    df_agg["col_key"] = (df_agg["model"] + " | "
                         + df_agg["dataset"] + f" ({agg})")

    # ── Step 3: pivot ─────────────────────────────────────────────────────────
    wide = df_agg.pivot(index="feature",
                        columns="col_key",
                        values=f"icc_{agg}")

    # ── Step 4: deterministic column order ────────────────────────────────────
    # Group by model, then by split order: all → wikitext → fineweb
    split_order = ["all", "wikitext", "fineweb"]
    models_seen = dict.fromkeys(df_agg["model"].tolist())      # preserves order
    col_order   = [
        f"{m} | {ds} ({agg})"
        for m  in models_seen
        for ds in split_order
        if f"{m} | {ds} ({agg})" in wide.columns
    ]
    wide = wide[col_order]

    # ── Step 5: add qualitative rating column for each model ("all" split) ───
    rating_cols = {}
    for col in col_order:
        if f"| all ({agg})" in col:
            model_label = col.replace(f" | all ({agg})", "")
            rating_col  = f"{model_label} | rating ({agg})"
            rating_cols[col] = rating_col
            wide[rating_col] = wide[col].apply(_rate)

    # Interleave rating columns right after their ICC column
    final_order = []
    for col in col_order:
        final_order.append(col)
        if col in rating_cols:
            final_order.append(rating_cols[col])
    wide = wide[final_order]

    # ── Step 6: sort rows by first "all" ICC column descending ────────────────
    sort_col = next(c for c in col_order if f"| all ({agg})" in c)
    wide = wide.sort_values(sort_col, ascending=False)

    return wide.round(4)


# =============================================================================
# CORRELATION MATRIX ANALYSIS — Multi-Model, Multi-Dataset
# Spearman / Pearson correlation across all taxonomy features.
#
# Granularity strategy:
#   model_dependent features : unique per (layer_idx, head_idx)      → df_A
#   input_dependent + target : unique per (layer_idx, head_idx, prompt_idx) → df_B
#
#   Correlation blocks:
#     model_dep × model_dep  → computed on df_A  (avoids artificial rank ties)
#     all other pairs        → computed on df_B  (broadcast of model_dep values)
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display


# Category display labels for plot annotations
CATEGORY_LABELS = {
    "model_dependent": "Model-dependent",
    "input_dependent": "Input-dependent",
    "target":          "Target",
}


# =============================================================================
# FEATURE HELPERS
# =============================================================================

def _get_all_features(taxonomy: dict,
                       df: pd.DataFrame,
                       exclude_subgroups: set | None = None) -> list[str]:
    """
    Returns an ordered, deduplicated list of features present in df,
    respecting taxonomy order. Subgroups listed in exclude_subgroups are skipped.
    """
    exclude_subgroups = exclude_subgroups or set()
    seen, feats = set(), []
    for category in taxonomy.values():
        for subgroup_name, meta in category.items():
            if subgroup_name in exclude_subgroups:
                continue
            for f in meta["features"]:
                if f in df.columns and f not in seen:
                    feats.append(f)
                    seen.add(f)
    return feats


def _feat_to_category(taxonomy: dict) -> dict[str, str]:
    """Maps each feature name to its taxonomy category key."""
    mapping = {}
    for cat_name, subgroups in taxonomy.items():
        for meta in subgroups.values():
            for f in meta["features"]:
                mapping[f] = cat_name
    return mapping


# =============================================================================
# CORRELATION MATRIX BUILDER
# =============================================================================

def _build_corr_df(df: pd.DataFrame,
                    features: list[str],
                    model_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Constructs two DataFrames at the appropriate granularity:

    df_A : (layer_idx, head_idx) — model-dependent features only.
           Deduplication uses FEATURE_DEDUP keys; kv_head derived from
           model_cfg["gqa_ratio"] (no hardcoded // 4).

    df_B : (layer_idx, head_idx, prompt_idx) — input-dependent + target features,
           with model-dependent values broadcast onto every prompt row.

    Returns (df_A, df_B). df_B is None when no input-dependent/target features exist.
    """
    base_idx   = ["layer_idx", "head_idx"]
    prompt_col = "prompt_idx" if "prompt_idx" in df.columns else None
    gqa_ratio  = model_cfg["gqa_ratio"]

    model_dep_feats = [f for f in features
                       if FEATURE_DEDUP.get(f) is not None and f in df.columns]
    other_feats     = [f for f in features
                       if FEATURE_DEDUP.get(f) is None     and f in df.columns]

    # ── df_A: deduplicated (layer, head) ─────────────────────────────────────
    df_A = (df[base_idx].drop_duplicates()
                         .sort_values(base_idx)
                         .reset_index(drop=True))

    for f in model_dep_feats:
        dedup = FEATURE_DEDUP[f]
        tmp   = df.copy()

        if "kv_head" in dedup:
            if "kv_head" not in tmp.columns:
                tmp["kv_head"] = tmp["head_idx"] // gqa_ratio
            mapping = (tmp.drop_duplicates(subset=["layer_idx", "kv_head"])
                          [["layer_idx", "kv_head", f]])
            df_A["kv_head"] = df_A["head_idx"] // gqa_ratio
            df_A = df_A.merge(mapping, on=["layer_idx", "kv_head"], how="left")
        else:
            mapping = (tmp.drop_duplicates(subset=dedup)
                          [[c for c in dedup if c in tmp.columns] + [f]])
            df_A = df_A.merge(mapping,
                              on=[c for c in dedup if c in tmp.columns],
                              how="left")

    df_A = df_A.drop(columns=["kv_head"], errors="ignore").set_index(base_idx)

    # ── df_B: full (layer, head, prompt) with broadcast ──────────────────────
    if not other_feats:
        return df_A, None

    prompt_cols = base_idx + ([prompt_col] if prompt_col else []) + other_feats
    df_B = df[[c for c in prompt_cols if c in df.columns]].copy().reset_index(drop=True)

    # Broadcast model-dependent values onto every prompt row
    for f in model_dep_feats:
        df_B = df_B.merge(df_A[[f]].reset_index(), on=base_idx, how="left")

    index_cols = base_idx + ([prompt_col] if prompt_col else [])
    df_B = df_B.set_index(index_cols)

    return df_A, df_B


def _build_corr_matrix(df_A: pd.DataFrame,
                        df_B: pd.DataFrame | None,
                        features: list[str],
                        method: str) -> pd.DataFrame:
    """
    Assembles the full correlation matrix by block:
      - model_dep × model_dep  : from df_A (granularity: layer × head)
      - all other pairs        : from df_B (granularity: layer × head × prompt)

    The split avoids the artificial Spearman = 1.0 artifact that arises when
    model-dependent constants are repeated across prompt rows.
    """
    model_dep = [f for f in features if FEATURE_DEDUP.get(f) is not None]
    feat_idx  = {f: i for i, f in enumerate(features)}
    n         = len(features)
    corr_mat  = np.full((n, n), np.nan)

    # Block 1: model_dep × model_dep
    if model_dep and df_A is not None:
        avail = [f for f in model_dep if f in df_A.columns]
        if len(avail) >= 2:
            c = df_A[avail].dropna(how="all").corr(method=method)
            for fi in avail:
                for fj in avail:
                    corr_mat[feat_idx[fi], feat_idx[fj]] = c.loc[fi, fj]

    # Block 2: all other pairs (including cross model_dep × other)
    if df_B is not None:
        avail = [f for f in features if f in df_B.columns]
        if len(avail) >= 2:
            c = df_B[avail].dropna(how="all").corr(method=method)
            for fi in avail:
                for fj in avail:
                    i, j = feat_idx[fi], feat_idx[fj]
                    # Do not overwrite the more accurate model_dep × model_dep block
                    if not (fi in model_dep and fj in model_dep):
                        corr_mat[i, j] = c.loc[fi, fj]

    # Enforce symmetry
    for i in range(n):
        for j in range(i):
            if np.isnan(corr_mat[i, j]) and not np.isnan(corr_mat[j, i]):
                corr_mat[i, j] = corr_mat[j, i]
            elif not np.isnan(corr_mat[i, j]) and np.isnan(corr_mat[j, i]):
                corr_mat[j, i] = corr_mat[i, j]
            else:
                v = np.nanmean([corr_mat[i, j], corr_mat[j, i]])
                corr_mat[i, j] = corr_mat[j, i] = v

    np.fill_diagonal(corr_mat, 1.0)
    return pd.DataFrame(corr_mat, index=features, columns=features)


# =============================================================================
# PLOTLY HEATMAP RENDERER
# =============================================================================

def _make_plotly_heatmap(corr: pd.DataFrame,
                          title: str,
                          feat_order: list[str],
                          feat_to_cat: dict[str, str]) -> go.Figure:
    """
    Lower-triangular interactive heatmap (Plotly).
    Black separator lines mark category boundaries.
    Hover shows both feature names and the Spearman r value.
    """
    n = len(feat_order)

    # Mask upper triangle
    z = corr.values.copy().astype(float)
    for i in range(n):
        for j in range(i + 1, n):
            z[i, j] = np.nan

    # Hover text
    hover = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if j > i:
                hover[i, j] = ""
            else:
                v = corr.values[i, j]
                hover[i, j] = (
                    f"<b>{feat_order[i]}</b><br>"
                    f"<b>{feat_order[j]}</b><br>"
                    f"r = {v:.3f}"
                ) if not np.isnan(v) else ""

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z,
        x=feat_order,
        y=feat_order,
        zmin=-1, zmax=1,
        colorscale="RdBu_r",
        zmid=0,
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        showscale=True,
        colorbar=dict(title="r", thickness=14, len=0.7),
        xgap=0.5, ygap=0.5,
    ))

    # Category boundary lines (confined to lower triangle)
    shapes = []
    current_cat = feat_to_cat.get(feat_order[0])
    for i, f in enumerate(feat_order[1:], start=1):
        cat = feat_to_cat.get(f)
        if cat != current_cat:
            shapes += [
                dict(type="line",                         # vertical — from diagonal down
                     x0=i-0.5, x1=i-0.5,
                     y0=i-0.5, y1=n-0.5,
                     line=dict(color="black", width=1.5), layer="above"),
                dict(type="line",                         # horizontal — from left to diagonal
                     x0=-0.5,  x1=i-0.5,
                     y0=i-0.5, y1=i-0.5,
                     line=dict(color="black", width=1.5), layer="above"),
            ]
            current_cat = cat

    # Category label annotations (above x-axis)
    annotations = []
    segments, seg_start, seg_cat = [], 0, feat_to_cat.get(feat_order[0])
    for i, f in enumerate(feat_order[1:], start=1):
        cat = feat_to_cat.get(f)
        if cat != seg_cat:
            segments.append((seg_start, i, seg_cat))
            seg_start, seg_cat = i, cat
    segments.append((seg_start, n, seg_cat))

    for s, e, cat in segments:
        annotations.append(dict(
            x=(s + e - 1) / 2, y=n + 0.8,
            text=f"<b>{CATEGORY_LABELS.get(cat, cat)}</b>",
            showarrow=False,
            font=dict(size=10, color="#333"),
            xref="x", yref="y",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#222"), x=0.5),
        xaxis=dict(tickangle=45, tickfont=dict(size=8), side="bottom",
                   constrain="domain", scaleanchor="y"),
        yaxis=dict(tickfont=dict(size=8), autorange="reversed", constrain="domain"),
        shapes=shapes,
        annotations=annotations,
        width=820, height=760,
        margin=dict(l=120, r=60, t=80, b=150),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# =============================================================================
# ENTRY POINT
# =============================================================================
import plotly.graph_objects as go


def _make_plotly_heatmap_rect(
    corr:        pd.DataFrame,
    title:       str,
    feat_to_cat: dict,
    cell_h:      int = 38,    # altezza pixel per cella (asse Y)
    cell_w:      int = 52,    # larghezza pixel per cella (asse X)
    font_size:   int = 10,    # font dentro le celle
    show_text:   bool = True, # mostra i valori numerici nelle celle
) -> "go.Figure":
    """
    Heatmap rettangolare (predictors × targets) con layout ottimizzato
    per leggibilità e salvataggio PNG per tesi.
    """
    row_feats = list(corr.index)
    col_feats = list(corr.columns)
    n_r = len(row_feats)
    n_c = len(col_feats)

    z    = corr.values
    text = [
        [f"{v:+.2f}" if not np.isnan(v) else "—" for v in row]
        for row in z
    ]

    # Colore testo adattivo: bianco su celle scure, nero su celle chiare
    textcolor = [
        ["white" if abs(v) > 0.5 else "black" for v in row]
        for row in z
    ]

    # ── Dimensioni dinamiche ───────────────────────────────────────────────────
    margin_l = 220   # spazio per label Y (nomi feature predictor, spesso lunghi)
    margin_b = 200   # spazio per label X ruotate (-45°)
    margin_t = 100   # titolo
    margin_r = 120   # colorbar

    plot_w = n_c * cell_w + margin_l + margin_r
    plot_h = n_r * cell_h + margin_t + margin_b

    # ── Heatmap trace ──────────────────────────────────────────────────────────
    heatmap = go.Heatmap(
        z=z,
        x=col_feats,
        y=row_feats,
        text=text if show_text else None,
        texttemplate="%{text}" if show_text else None,
        textfont=dict(size=font_size, color="black"),  # override sotto
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title=dict(text="ρ", font=dict(size=13)),
            thickness=18,
            len=0.85,
            tickfont=dict(size=11),
        ),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>ρ = %{z:.3f}<extra></extra>",
        xgap=1,  # piccolo gap tra le celle per separazione visiva
        ygap=1,
    )

    fig = go.Figure(heatmap)

    # ── Testo adattivo (bianco/nero) via annotations ───────────────────────────
    # Plotly non supporta textcolor per cella nel Heatmap → usiamo annotations
    if show_text:
        annotations = []
        for i, row_f in enumerate(row_feats):
            for j, col_f in enumerate(col_feats):
                val = z[i, j]
                if np.isnan(val):
                    continue
                color = "white" if abs(val) > 0.50 else "#111111"
                annotations.append(dict(
                    x=col_f,
                    y=row_f,
                    text=f"{val:+.2f}",
                    showarrow=False,
                    font=dict(size=font_size, color=color),
                    xref="x",
                    yref="y",
                ))
        fig.update_layout(annotations=annotations)

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, family="Arial"),
            x=0.0,
            xanchor="left",
            pad=dict(l=margin_l),
        ),
        width=plot_w,
        height=plot_h,
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=11, family="Arial"),
            side="bottom",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=11, family="Arial"),
            autorange="reversed",   # prima riga in alto
            showgrid=False,
            zeroline=False,
        ),
    )

    return fig

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path


def save_cross_heatmap_png(
    corr:       pd.DataFrame,
    title:      str,
    out_path:   str | Path,
    cell_h_in:  float = 0.45,   # altezza per cella in pollici
    cell_w_in:  float = 0.55,   # larghezza per cella in pollici
    fontsize:   int   = 9,
    dpi:        int   = 200,
    cmap:       str   = "RdBu_r",
    vmin:       float = -1.0,
    vmax:       float = 1.0,
) -> None:
    """
    Salva la matrice rettangolare predictors×targets come PNG usando Matplotlib.
    Matplotlib è deterministico: ciò che vedi è ciò che viene salvato.
    """
    row_feats = list(corr.index)
    col_feats = list(corr.columns)
    n_r = len(row_feats)
    n_c = len(col_feats)

    # Dimensioni figura dinamiche basate sul numero di celle
    label_y_in = 2.8   # spazio a sinistra per label feature Y (lunghe)
    label_x_in = 1.8   # spazio in basso per label feature X ruotate
    cbar_in    = 0.6   # spazio a destra per la colorbar
    title_in   = 0.5   # spazio in alto per il titolo

    fig_w = label_y_in + n_c * cell_w_in + cbar_in
    fig_h = title_in   + n_r * cell_h_in + label_x_in

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm   = plt.get_cmap(cmap)
    z    = corr.values

    # ── Disegna le celle ───────────────────────────────────────────────────────
    im = ax.imshow(z, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")

    # ── Valori numerici dentro le celle ────────────────────────────────────────
    for i in range(n_r):
        for j in range(n_c):
            val = z[i, j]
            if np.isnan(val):
                continue
            # Colore testo adattivo: bianco su celle scure
            bg_rgba  = cm(norm(val))
            luminance = 0.299*bg_rgba[0] + 0.587*bg_rgba[1] + 0.114*bg_rgba[2]
            txt_color = "white" if luminance < 0.50 else "#111111"
            ax.text(j, i, f"{val:+.2f}",
                    ha="center", va="center",
                    fontsize=fontsize, color=txt_color,
                    fontfamily="DejaVu Sans")

    # ── Assi ──────────────────────────────────────────────────────────────────
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(col_feats, rotation=45, ha="right",
                       fontsize=fontsize + 1)

    ax.set_yticks(range(n_r))
    ax.set_yticklabels(row_feats, fontsize=fontsize + 1)

    ax.tick_params(axis="both", which="both", length=0)  # togli i tick marks

    # Linee di separazione tra celle
    ax.set_xticks(np.arange(-0.5, n_c, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_r, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("ρ", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # ── Titolo ────────────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=11, fontweight="bold",
                 pad=10, loc="left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] saved → {out_path}")

def plot_correlation_matrices(
    df_combined:       pd.DataFrame,
    model_keys:        list[str] | None = None,
    taxonomy:          dict             = TAXONOMY,
    method:            str              = "spearman",
    exclude_subgroups: set | None       = None,
    splits:            list[str] | None = None,
    save:              bool             = SAVE_PLOTS,
    png_scale:         float            = 2.0,
    px_per_row:        int              = 35,
    px_per_col:        int              = 45,
) -> dict[str, dict[str, tuple]]:

    if model_keys is None:
        model_keys = list(MODEL_CONFIGS.keys())
    if splits is None:
        splits = ["all", "wikitext", "fineweb"]

    target_feats: list[str] = []
    if "target" in taxonomy:
        for group in taxonomy["target"].values():
            target_feats.extend(group["features"])

    predictor_feats: list[str] = []
    for sec in ("model_dependent", "input_dependent"):
        if sec in taxonomy:
            for group in taxonomy[sec].values():
                predictor_feats.extend(group["features"])

    results: dict[str, dict] = {}

    for model_key in model_keys:
        cfg      = MODEL_CONFIGS[model_key]
        df_model = df_combined[df_combined["model_name"] == cfg["model_name"]].copy()

        if df_model.empty:
            print(f"[WARNING] No data for model '{model_key}'. Skipping.")
            continue

        features    = _get_all_features(taxonomy, df_model, exclude_subgroups)
        feat_to_cat = _feat_to_category(taxonomy)
        results[model_key] = {}

        model_dir = out_dir / cfg["model_name"] if save else None

        print(f"\n── {cfg['label']} — {len(features)} features ──")

        split_filters = {
            "all":      df_model,
            "wikitext": df_model[df_model["prompt_source"] == cfg["ptype_wiki"]],
            "fineweb":  df_model[df_model["prompt_source"] == cfg["ptype_fineweb"]],
        }

        for split in splits:
            df_split = split_filters.get(split, df_model)
            if df_split.empty:
                print(f"  [SKIP] split '{split}' — no data.")
                continue

            # ── Full matrix ───────────────────────────────────────────────────
            df_A, df_B = _build_corr_df(df_split, features, cfg)
            corr_full  = _build_corr_matrix(df_A, df_B, features, method)

            title_full = (
                f"Correlation Matrix — {cfg['label']}<br>"
                f"Split: {split}  |  Method: {method}"
            )
            fig_full = _make_plotly_heatmap(corr_full, title_full, features, feat_to_cat)

            if save:
                model_dir.mkdir(parents=True, exist_ok=True)
                fig_full.write_image(
                    str(model_dir / f"corr_{split}_{method}.png"),
                    width=1200,
                    height=1200,
                    scale=png_scale,
                )

            fig_full.show()
            print(f"  › {split} — full matrix ({corr_full.shape[0]}×{corr_full.shape[1]})")

            # ── Cross-correlation block (predictors × targets) ─────────────────
            valid_x    = [f for f in target_feats    if f in corr_full.columns]
            valid_y    = [f for f in predictor_feats if f in corr_full.index]
            corr_cross = corr_full.loc[valid_y, valid_x] if (valid_x and valid_y) else None

            results[model_key][split] = (corr_full, corr_cross)

            if corr_cross is not None:
                n_rows, n_cols = corr_cross.shape

                # Titolo con \n per matplotlib, <br> per Plotly
                title_cross      = f"Predictors × Targets — {cfg['label']}\nSplit: {split}  |  Method: {method}"
                title_cross_html = title_cross.replace("\n", "<br>")

                fig_cross = _make_plotly_heatmap_rect(corr_cross, title_cross_html, feat_to_cat)
                fig_cross.show()

                if save:
                    model_dir.mkdir(parents=True, exist_ok=True)
                    save_cross_heatmap_png(
                        corr_cross,
                        title=title_cross,
                        out_path=model_dir / f"corr_{split}_{method}_CROSS.png",
                    )

                print(f"  › {split} — cross block ({n_rows} predictors × {n_cols} targets)")
            else:
                print(f"  [SKIP] {split} — cross block: no matching features.")

    return results


# =============================================================================
# CORRELATION MATRIX ANALYSIS — By Layer Quartile, Multi-Model
#
# Partitions layers into 4 depth quartiles (Q1=early … Q4=late) and computes
# a Spearman/Pearson correlation matrix per quartile, optionally split by
# dataset. Visualises each matrix as an interactive lower-triangular heatmap.
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display


# =============================================================================
# INTERNAL: quartile partitioning
# =============================================================================

def _build_quartile_map(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Assigns each layer_idx to one of four depth quartiles (0–3).

    Returns
    -------
    layer_to_q : dict  layer_idx → quartile index (0-3)
    q_labels   : dict  quartile index → human-readable label
    """
    layers   = sorted(df["layer_idx"].unique())
    n_layers = len(layers)
    q_size   = n_layers // 4

    layer_to_q = {l: min(i // q_size, 3) for i, l in enumerate(layers)}

    q_labels = {
        0: f"Q1 early  (L{layers[0]}–{layers[q_size - 1]})",
        1: f"Q2        (L{layers[q_size]}–{layers[2 * q_size - 1]})",
        2: f"Q3        (L{layers[2 * q_size]}–{layers[3 * q_size - 1]})",
        3: f"Q4 late   (L{layers[3 * q_size]}–{layers[-1]})",
    }
    return layer_to_q, q_labels


# =============================================================================
# ENTRY POINT
# =============================================================================

def plot_correlation_by_quartile(
    df_combined: pd.DataFrame,
    model_keys: list[str] | None = None,
    taxonomy: dict = TAXONOMY,
    method: str = "spearman",
    exclude_subgroups: set | None = None,
    split_datasets: bool = True,
    save: bool = SAVE_PLOTS,
) -> dict:
    """
    Computes and displays correlation heatmaps stratified by layer quartile.

    The iteration space is:
        M  (models)  ×  Q  (quartiles 0-3)  ×  D  (dataset splits, optional)

    Parameters
    ----------
    df_combined      : Combined DataFrame with a "model_name" column.
    model_keys       : Subset of MODEL_CONFIGS keys. Defaults to all.
    taxonomy         : Feature taxonomy dict. Defaults to module-level TAXONOMY.
    method           : "spearman" (default) or "pearson".
    exclude_subgroups: Taxonomy subgroup names to exclude (e.g. {"RMSNorm"}).
    split_datasets   : If True, computes separate matrices for wikitext and
                       fineweb. If False, uses the combined data for each quartile.
    save             : If True, writes HTML files to out_dir/<model_name>/.

    Returns
    -------
    Nested dict:
        results[model_key][(q_idx, split_label)] = pd.DataFrame (corr matrix)
    """
    if model_keys is None:
        model_keys = list(MODEL_CONFIGS.keys())

    results = {}

    for model_key in model_keys:
        cfg      = MODEL_CONFIGS[model_key]
        df_model = df_combined[df_combined["model_name"] == cfg["model_name"]].copy()

        if df_model.empty:
            print(f"[WARNING] No data for model '{model_key}'. Skipping.")
            continue

        features    = _get_all_features(taxonomy, df_model, exclude_subgroups)
        feat_to_cat = _feat_to_category(taxonomy)

        layer_to_q, q_labels = _build_quartile_map(df_model)
        df_model["_quartile"] = df_model["layer_idx"].map(layer_to_q)

        # Build the list of (label, df_subset) pairs to iterate over
        if split_datasets:
            splits = [
                ("wikitext", df_model[df_model["prompt_source"] == cfg["ptype_wiki"]]),
                ("fineweb",  df_model[df_model["prompt_source"] == cfg["ptype_fineweb"]]),
            ]
        else:
            splits = [("all", df_model)]

        results[model_key] = {}

        print(f"\n{'═' * 64}")
        print(f"  Model   : {cfg['label']}")
        print(f"  Method  : {method}  |  Dataset split: {split_datasets}")
        print(f"  Layers  : {df_model['layer_idx'].nunique()}  "
              f"→  {len(q_labels)} quartiles × {len(splits)} split(s)")
        print(f"{'═' * 64}")

        for split_label, df_split in splits:
            if df_split.empty:
                print(f"  [SKIP] split '{split_label}' — no data.")
                continue

            for q_idx in range(4):
                df_q = df_split[df_split["_quartile"] == q_idx]

                if df_q.empty:
                    print(f"  [SKIP] {split_label} / {q_labels[q_idx]} — empty.")
                    continue

                df_A, df_B = _build_corr_df(df_q, features, cfg)

                n_A = df_A.shape[0]
                n_B = df_B.shape[0] if df_B is not None else 0
                if n_A < 3 and n_B < 3:
                    print(f"  [SKIP] {split_label} / {q_labels[q_idx]} — "
                          f"insufficient observations (A={n_A}, B={n_B}).")
                    continue

                corr       = _build_corr_matrix(df_A, df_B, features, method)
                feats_ord  = [f for f in features if f in corr.columns]
                corr       = corr.loc[feats_ord, feats_ord]

                split_tag  = f"  |  {split_label}" if split_datasets else ""
                title      = (f"Correlation — {cfg['label']}<br>"
                              f"{q_labels[q_idx]}{split_tag}  |  {method}")
                fig        = _make_plotly_heatmap(corr, title, feats_ord, feat_to_cat)

                key = (q_idx, split_label)
                results[model_key][key] = corr

                if save:
                    model_dir = out_dir / cfg["model_name"]
                    model_dir.mkdir(exist_ok=True)
                    safe = (f"corr_q{q_idx}_{split_label}_{method}"
                            .replace(" ", "_"))
                    fig.write_html(str(model_dir / f"{safe}.html"))

                print(f"  › Q{q_idx + 1}  {split_label:<10}  "
                      f"rows A={n_A}  B={n_B}")
                fig.show()

    return results
def plot_cross_correlation_by_quartile(
    df_combined: pd.DataFrame,
    model_keys: list[str] | None = None,
    taxonomy: dict = TAXONOMY,
    method: str = "spearman",
    exclude_subgroups: set | None = None,
    split_datasets: bool = True,
    save: bool = SAVE_PLOTS,
) -> dict:
    """
    Computes and displays rectangular cross-correlation heatmaps 
    (Predictors × Targets) stratified by layer quartile.
    """
    if model_keys is None:
        model_keys = list(MODEL_CONFIGS.keys())
        
    # Se non viene passato un set specifico, escludiamo di default il sottogruppo RMSNorm
    if exclude_subgroups is None:
        exclude_subgroups = {"RMSNorm"}
    else:
        exclude_subgroups.add("RMSNorm")

    # Pre-calcoliamo la lista delle feature target e dei predittori
    target_feats: list[str] = []
    if "target" in taxonomy:
        for group in taxonomy["target"].values():
            if group.get("label", "") not in exclude_subgroups:
                target_feats.extend(group["features"])

    predictor_feats: list[str] = []
    for sec in ("model_dependent", "input_dependent"):
        if sec in taxonomy:
            for group_name, group in taxonomy[sec].items():
                if group_name not in exclude_subgroups and group.get("label", "") not in exclude_subgroups:
                    predictor_feats.extend(group["features"])

    results = {}

    for model_key in model_keys:
        cfg      = MODEL_CONFIGS[model_key]
        df_model = df_combined[df_combined["model_name"] == cfg["model_name"]].copy()

        if df_model.empty:
            print(f"[WARNING] No data for model '{model_key}'. Skipping.")
            continue

        features    = _get_all_features(taxonomy, df_model, exclude_subgroups)
        feat_to_cat = _feat_to_category(taxonomy)

        layer_to_q, q_labels = _build_quartile_map(df_model)
        df_model["_quartile"] = df_model["layer_idx"].map(layer_to_q)

        # Build the list of (label, df_subset) pairs to iterate over
        if split_datasets:
            splits = [
                ("wikitext", df_model[df_model["prompt_source"] == cfg["ptype_wiki"]]),
                ("fineweb",  df_model[df_model["prompt_source"] == cfg["ptype_fineweb"]]),
            ]
        else:
            splits = [("all", df_model)]

        results[model_key] = {}

        print(f"\n{'═' * 64}")
        print(f"  Model   : {cfg['label']}")
        print(f"  Cross   : Predictors × Targets")
        print(f"  Method  : {method}  |  Dataset split: {split_datasets}")
        print(f"  Layers  : {df_model['layer_idx'].nunique()}  "
              f"→  {len(q_labels)} quartiles × {len(splits)} split(s)")
        print(f"{'═' * 64}")

        for split_label, df_split in splits:
            if df_split.empty:
                print(f"  [SKIP] split '{split_label}' — no data.")
                continue

            for q_idx in range(4):
                df_q = df_split[df_split["_quartile"] == q_idx]

                if df_q.empty:
                    print(f"  [SKIP] {split_label} / {q_labels[q_idx]} — empty.")
                    continue

                # Calcolo matrice full per l'intero quartile
                df_A, df_B = _build_corr_df(df_q, features, cfg)

                n_A = df_A.shape[0]
                n_B = df_B.shape[0] if df_B is not None else 0
                if n_A < 3 and n_B < 3:
                    print(f"  [SKIP] {split_label} / {q_labels[q_idx]} — "
                          f"insufficient observations (A={n_A}, B={n_B}).")
                    continue

                corr_full = _build_corr_matrix(df_A, df_B, features, method)
                
                # Ritaglio blocco Predictors × Targets
                valid_x    = [f for f in target_feats    if f in corr_full.columns]
                valid_y    = [f for f in predictor_feats if f in corr_full.index]
                
                if not valid_x or not valid_y:
                    print(f"  [SKIP] {split_label} / {q_labels[q_idx]} — no matching cross features.")
                    continue
                    
                corr_cross = corr_full.loc[valid_y, valid_x]

                split_tag  = f"  |  {split_label}" if split_datasets else ""
                title      = (f"Predictors × Targets — {cfg['label']}<br>"
                              f"{q_labels[q_idx]}{split_tag}  |  {method}")
                
                # Uso la funzione Plotly specifica per le rettangolari
                fig = _make_plotly_heatmap_rect(corr_cross, title, feat_to_cat)

                key = (q_idx, split_label)
                results[model_key][key] = corr_cross

                if save:
                    model_dir = out_dir / cfg["model_name"]
                    model_dir.mkdir(exist_ok=True)
                    safe = (f"corr_cross_q{q_idx}_{split_label}_{method}"
                            .replace(" ", "_"))
                    fig.write_html(str(model_dir / f"{safe}.html"))
                    
                    # Salva anche in PNG per tesi usando matplotlib
                    save_cross_heatmap_png(
                        corr_cross,
                        title=title.replace("<br>", "\n"),
                        out_path=model_dir / f"{safe}.png"
                    )

                n_rows, n_cols = corr_cross.shape
                print(f"  › Q{q_idx + 1}  {split_label:<10}  "
                      f"rows A={n_A}  B={n_B}  |  cross={n_rows}×{n_cols}")
                fig.show()

    return results

# =============================================================================
# PAIRWISE SCATTER ANALYSIS — Model-dependent (X) × Target (Y)
# Stratified by layer quartile, multi-model.
# v2: improved layout — no title/plot overlap, clean colorbar, safe legend.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import theilslopes
from IPython.display import display


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _select_quartile_layers(df: pd.DataFrame, quartile: int) -> list:
    layers = sorted(df["layer_idx"].unique())
    n      = len(layers)
    q_size = n // 4
    q_start = quartile * q_size
    q_end   = (quartile + 1) * q_size if quartile < 3 else n
    return layers[q_start:q_end]


def _build_plot_df(df_q: pd.DataFrame,
                   x_feats: list[str],
                   y_feats: list[str],
                   model_cfg: dict,
                   agg: str | None) -> pd.DataFrame:
    all_feats  = list(dict.fromkeys(x_feats + y_feats))
    df_A, df_B = _build_corr_df(df_q, all_feats, model_cfg)

    if df_B is None:
        return df_A.reset_index()

    plot_df = df_B.reset_index()

    if agg is not None:
        agg_fn   = "median" if agg == "median" else "mean"
        base_idx = ["layer_idx", "head_idx"]
        drop_cols = [c for c in ["prompt_idx"] if c in plot_df.columns]
        num_cols  = [c for c in plot_df.columns
                     if c not in base_idx + drop_cols
                     and pd.api.types.is_numeric_dtype(plot_df[c])]
        plot_df = (plot_df
                   .groupby(base_idx)[num_cols]
                   .agg(agg_fn)
                   .reset_index())

    md_in_A = [f for f in x_feats if f in df_A.columns]
    if md_in_A:
        plot_df = plot_df.merge(
            df_A[md_in_A].reset_index(),
            on=["layer_idx", "head_idx"],
            how="left",
            suffixes=("", "_md"),
        )
        for f in md_in_A:
            if f"{f}_md" in plot_df.columns:
                plot_df[f] = plot_df[f"{f}_md"]
                plot_df    = plot_df.drop(columns=[f"{f}_md"])

    return plot_df


def _draw_cell(ax, sub: pd.DataFrame, x_f: str, y_f: str,
               cmap, norm, method: str,
               show_xlabel: bool, show_ylabel: bool) -> None:
    if sub.empty:
        ax.set_visible(False)
        return

    ax.scatter(
        sub[x_f], sub[y_f],
        c=sub["layer_idx"],
        cmap=cmap, norm=norm,
        s=14, alpha=0.60, linewidths=0, rasterized=True,
    )

    if len(sub) > 5:
        try:
            res   = theilslopes(sub[y_f].values, sub[x_f].values)
            x_fit = np.array([sub[x_f].min(), sub[x_f].max()])
            ax.plot(x_fit, res.slope * x_fit + res.intercept,
                    color="#cc2222", lw=1.2, ls="--", zorder=4, alpha=0.90)
        except Exception:
            pass

        rho        = sub[[x_f, y_f]].corr(method=method).iloc[0, 1]
        ann_color  = "#cc2222" if abs(rho) > 0.4 else "#444444"
        ann_weight = "bold"    if abs(rho) > 0.4 else "normal"
        ax.annotate(
            f"ρ = {rho:.2f}",
            xy=(0.05, 0.88), xycoords="axes fraction",
            fontsize=7, color=ann_color, fontweight=ann_weight,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.82),
            zorder=5,
        )

    # Axis labels — short names to avoid overlap
    if show_xlabel:
        label = x_f.replace("rope_", "").replace("_", " ")
        ax.set_xlabel(label, fontsize=7, labelpad=3)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")

    if show_ylabel:
        label = y_f.replace("diagonal_mass", "diag").replace("_", " ")
        ax.set_ylabel(label, fontsize=7, labelpad=3)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    ax.tick_params(labelsize=6, length=3, width=0.5)
    ax.grid(True, lw=0.25, alpha=0.30, color="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#bbbbbb")


# =============================================================================
# ENTRY POINT
# =============================================================================

def plot_scatter_quartile(
    df_combined: pd.DataFrame,
    model_keys: list[str] | None = None,
    quartile: int = 0,
    x_feats: list[str] | None = None,
    y_feats: list[str] | None = None,
    method: str = "spearman",
    agg: str | None = "median",
    dataset: str = "all",
    save: bool = SAVE_PLOTS,
) -> None:
    """
    Pairwise scatter grid: x_feats (model-dependent) × y_feats (target).
    One figure per model.

    Parameters
    ----------
    df_combined : Combined DataFrame with "model_name" column.
    model_keys  : Models to plot. Defaults to all in MODEL_CONFIGS.
    quartile    : Layer depth quartile — 0=Q1 (early) … 3=Q4 (late).
    x_feats     : Model-dependent features for X axes. Defaults to DEFAULT_X_FEATS.
    y_feats     : Target features for Y axes. Defaults to DEFAULT_Y_FEATS.
    method      : Correlation method — "spearman" or "pearson".
    agg         : Point aggregation over prompts:
                    None       → raw (layer, head, prompt)
                    "median"   → per-(layer, head) median  [default]
                    "mean"     → per-(layer, head) mean
    dataset     : "all" | "wikitext" | "fineweb"
    save        : Saves PNG to out_dir/<model_name>/ if True.
    """
    assert quartile in (0, 1, 2, 3),          "quartile must be 0-3."
    assert agg in (None, "median", "mean"),    "agg must be None, 'median', or 'mean'."
    assert dataset in ("all", "wikitext", "fineweb"), \
        "dataset must be 'all', 'wikitext', or 'fineweb'."

    if model_keys is None:
        model_keys = list(MODEL_CONFIGS.keys())

    x_feats = x_feats or DEFAULT_X_FEATS
    y_feats = y_feats or DEFAULT_Y_FEATS

    for model_key in model_keys:
        cfg      = MODEL_CONFIGS[model_key]
        df_model = df_combined[df_combined["model_name"] == cfg["model_name"]].copy()
        if df_model.empty:
            print(f"[WARNING] No data for '{model_key}'. Skipping.")
            continue

        # ── Dataset filter ────────────────────────────────────────────────────
        if dataset == "wikitext":
            df_filt = df_model[df_model["prompt_source"] == cfg["ptype_wiki"]]
        elif dataset == "fineweb":
            df_filt = df_model[df_model["prompt_source"] == cfg["ptype_fineweb"]]
        else:
            df_filt = df_model

        # ── Quartile filter ───────────────────────────────────────────────────
        q_layers = _select_quartile_layers(df_filt, quartile)
        df_q     = df_filt[df_filt["layer_idx"].isin(q_layers)].copy()
        q_label  = f"Q{quartile + 1}  (L{q_layers[0]}–{q_layers[-1]})"

        if df_q.empty:
            print(f"[SKIP] {cfg['label']} / {q_label} — empty.")
            continue

        # ── Validate features ─────────────────────────────────────────────────
        x_valid = [f for f in x_feats if f in df_q.columns]
        y_valid = [f for f in y_feats if f in df_q.columns]
        missing = set(x_feats + y_feats) - set(x_valid + y_valid)
        if missing:
            print(f"  [INFO] {cfg['label']}: missing — {sorted(missing)}")
        if not x_valid or not y_valid:
            print(f"  [SKIP] {cfg['label']}: no valid features.")
            continue

        plot_df = _build_plot_df(df_q, x_valid, y_valid, cfg, agg)
        n_y, n_x = len(y_valid), len(x_valid)

        print(f"\n── {cfg['label']}  |  {q_label}  |  dataset={dataset}"
              f"  |  agg={agg or 'raw'}  |  n={len(plot_df):,}")

        # ── Color map ─────────────────────────────────────────────────────────
        cmap = cm.get_cmap("plasma", len(q_layers))
        norm = mcolors.Normalize(vmin=q_layers[0], vmax=q_layers[-1])

        # ── Figure: GridSpec with dedicated rows for title and legend ─────────
        # Layout:
        #   row 0       : suptitle spacer (height_ratio 0.18)
        #   rows 1..n_y : scatter grid
        #   last row    : legend spacer (height_ratio 0.12)
        # Colorbar is drawn as a separate axes on the right, outside the grid.

        cell_w   = 3.2          # inches per scatter cell
        cell_h   = 2.8          # inches per scatter cell
        cbar_w   = 0.55         # inches for colorbar
        margin_l = 0.9          # left margin for y-axis labels
        margin_r = 0.20         # right margin before colorbar
        margin_t = 1.00         # top margin for suptitle (generous)
        margin_b = 0.70         # bottom margin for legend + xlabel

        fig_w = margin_l + n_x * cell_w + margin_r + cbar_w + 0.3
        fig_h = margin_t + n_y * cell_h + margin_b

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=160)

        # Compute normalised margins for subplots_adjust
        left   = margin_l / fig_w
        right  = 1.0 - (margin_r + cbar_w + 0.3) / fig_w
        top    = 1.0 - margin_t / fig_h
        bottom = margin_b / fig_h

        # Main scatter grid
        gs = gridspec.GridSpec(
            n_y, n_x,
            left=left, right=right,
            top=top,   bottom=bottom,
            hspace=0.22, wspace=0.15,
        )
        axes = np.array([[fig.add_subplot(gs[r, c])
                          for c in range(n_x)]
                         for r in range(n_y)])

        # Colorbar axes — placed to the right of the grid
        cbar_left   = right + margin_r / fig_w
        cbar_bottom = bottom
        cbar_height = top - bottom
        cbar_width  = (cbar_w * 0.35) / fig_w
        cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

        # ── Draw cells ────────────────────────────────────────────────────────
        for row, y_f in enumerate(y_valid):
            for col, x_f in enumerate(x_valid):
                ax     = axes[row, col]
                needed = [c for c in [x_f, y_f, "layer_idx"] if c in plot_df.columns]
                sub    = plot_df[needed].dropna(subset=[x_f, y_f])
                _draw_cell(
                    ax, sub, x_f, y_f,
                    cmap=cmap, norm=norm, method=method,
                    show_xlabel=(row == n_y - 1),
                    show_ylabel=(col == 0),
                )

        # ── Colorbar ──────────────────────────────────────────────────────────
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Layer index", fontsize=8, labelpad=6)
        cbar.set_ticks(q_layers[::max(1, len(q_layers) // 6)])
        cbar.ax.tick_params(labelsize=7)

        # ── Legend — placed inside figure at bottom, anchored via fig coords ──
        legend_handles = [
            Line2D([0], [0], color="#cc2222", lw=1.2, ls="--",
                   label="|ρ| > 0.40 — Theil-Sen regression"),
            Line2D([0], [0], color="#444444", lw=1.2, ls="--",
                   label="|ρ| ≤ 0.40"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=2,
            fontsize=8,
            frameon=True,
            framealpha=0.92,
            edgecolor="#cccccc",
            bbox_to_anchor=(left + (right - left) / 2, 0.005),
            bbox_transform=fig.transFigure,
        )

        # ── Suptitle — anchored to top margin, never overlapping ──────────────
        agg_tag     = f"agg: {agg}" if agg else "raw points"
        dataset_tag = dataset
        fig.text(
            0.5, 1.0 - 0.15 / fig_h,          # just below top edge
            f"Model-dependent (X)  ×  Target (Y)",
            ha="center", va="top",
            fontsize=11, fontweight="bold", color="#111111",
            transform=fig.transFigure,
        )
        fig.text(
            0.5, 1.0 - 0.52 / fig_h,
            f"{cfg['label']}  |  {q_label}  |  {dataset_tag}  |  {agg_tag}",
            ha="center", va="top",
            fontsize=9, color="#333333",
            transform=fig.transFigure,
        )

        # ── Save / display ────────────────────────────────────────────────────
        if save:
            model_dir = out_dir / cfg["model_name"]
            model_dir.mkdir(exist_ok=True)
            safe_agg = agg or "raw"
            fname    = f"scatter_q{quartile+1}_{dataset}_{safe_agg}_{method}.png"
            fig.savefig(model_dir / fname, dpi=160, bbox_inches="tight")

        display(fig)
        plt.close(fig)



# =============================================================================
# LAYERWISE CORRELATION ANALYSIS — Model-dependent (X) × Target (Y)
# Computes per-layer Spearman/Pearson ρ between model-dependent features and
# target attention metrics, across all heads × prompts for each layer.
#
# Granularity:
#   X (model-dep) : one value per (layer, head) — deduplicated via FEATURE_DEDUP
#   Y (target)    : one value per (layer, head, prompt)
#   Join          : X broadcast onto Y rows via (layer, head)
#   Correlation   : computed over n_heads × n_prompts pairs per layer
#
# Output: pivot table — rows=(x_feat, y_feat), columns=layer_idx
# =============================================================================

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# =============================================================================
# DEFAULT FEATURE SETS (override at call time)
# =============================================================================

DEFAULT_LAYERWISE_X_FEATS = [
    "gini_left_Wq",          "gini_right_Wq",
    "gini_left_Wk",          "gini_right_Wk",
    "rope_pair_var_Wq",      "rope_pair_var_Wk",
    "rope_pair_max_ratio_Wq","rope_pair_max_ratio_Wk",
    "rope_freq_com_Wq",      "rope_freq_com_Wk",
]

DEFAULT_LAYERWISE_Y_FEATS = [
    "diagonal_mass_1",           "diagonal_mass_5",
    "diagonal_mass_1_shifted_1", "diagonal_mass_1_shifted_2",
    "diagonal_mass_1_shifted_3", "diagonal_mass_1_shifted_4",
]


# =============================================================================
# CORE COMPUTATION
# =============================================================================

def compute_layerwise_corr(
    df: pd.DataFrame,
    model_cfg: dict,
    x_feats: list[str] | None = None,
    y_feats: list[str] | None = None,
    method: str = "spearman",
    dataset: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes layer-wise pairwise correlation between model-dependent (X)
    and target (Y) features for a single model.

    Parameters
    ----------
    df         : DataFrame for a single model (already filtered by model_name).
    model_cfg  : Entry from MODEL_CONFIGS.
    x_feats    : Model-dependent features. Defaults to DEFAULT_LAYERWISE_X_FEATS.
    y_feats    : Target features. Defaults to DEFAULT_LAYERWISE_Y_FEATS.
    method     : "spearman" (default) or "pearson".
    dataset    : "all" | "wikitext" | "fineweb"

    Returns
    -------
    table   : pd.DataFrame — pivot  rows=(x_feat, y_feat), cols=layer_idx, values=ρ
    records : pd.DataFrame — long format with rho, pval, n_pairs per (layer, x, y)
    """
    x_feats = x_feats or DEFAULT_LAYERWISE_X_FEATS
    y_feats = y_feats or DEFAULT_LAYERWISE_Y_FEATS

    # ── Dataset filter ────────────────────────────────────────────────────────
    if dataset == "wikitext":
        df = df[df["prompt_source"] == model_cfg["ptype_wiki"]].copy()
    elif dataset == "fineweb":
        df = df[df["prompt_source"] == model_cfg["ptype_fineweb"]].copy()

    # ── Validate feature availability ─────────────────────────────────────────
    x_valid = [f for f in x_feats if f in df.columns]
    y_valid = [f for f in y_feats if f in df.columns]
    if not x_valid or not y_valid:
        raise ValueError(
            f"[{model_cfg['label']}] No valid features found. "
            f"Missing X: {set(x_feats)-set(x_valid)}, "
            f"Missing Y: {set(y_feats)-set(y_valid)}"
        )

    base_idx   = ["layer_idx", "head_idx"]
    prompt_col = "prompt_idx" if "prompt_idx" in df.columns else None
    layers     = sorted(df["layer_idx"].unique())
    gqa_ratio  = model_cfg["gqa_ratio"]
    records    = []

    for layer in layers:
        df_l = df[df["layer_idx"] == layer]

        # ── Spine: granularity (head, prompt) with Y features ─────────────────
        spine_cols = base_idx + ([prompt_col] if prompt_col else []) + y_valid
        spine = (df_l[[c for c in spine_cols if c in df_l.columns]]
                 .copy()
                 .reset_index(drop=True))

        # ── Broadcast X (model-dep) onto every spine row ──────────────────────
        for x_f in x_valid:
            dedup = FEATURE_DEDUP.get(x_f)
            tmp   = df_l.copy()

            if dedup is not None and "kv_head" in dedup:
                if "kv_head" not in tmp.columns:
                    tmp["kv_head"] = tmp["head_idx"] // gqa_ratio
                mapping = (tmp.drop_duplicates(subset=["layer_idx", "kv_head"])
                              [["head_idx", "kv_head", x_f]])
                spine["kv_head"] = spine["head_idx"] // gqa_ratio
                spine = (spine
                         .merge(mapping[["kv_head", x_f]], on="kv_head", how="left")
                         .drop(columns=["kv_head"]))
            else:
                mapping = (tmp.drop_duplicates(subset=base_idx)
                              [["head_idx", x_f]])
                spine = spine.merge(mapping, on="head_idx", how="left",
                                    suffixes=("", "_x"))
                # Resolve potential column collision from repeated merges
                if f"{x_f}_x" in spine.columns:
                    spine[x_f] = spine[f"{x_f}_x"]
                    spine = spine.drop(columns=[f"{x_f}_x"])

        # ── Per-pair correlation ───────────────────────────────────────────────
        for x_f in x_valid:
            if x_f not in spine.columns:
                continue
            for y_f in y_valid:
                if y_f not in spine.columns:
                    continue

                sub = spine[[x_f, y_f]].dropna()

                if len(sub) < 5:
                    rho, pval = np.nan, np.nan
                elif method == "spearman":
                    rho, pval = spearmanr(sub[x_f].values, sub[y_f].values)
                elif method == "pearson":
                    from scipy.stats import pearsonr
                    rho, pval = pearsonr(sub[x_f].values, sub[y_f].values)
                else:
                    raise ValueError(f"Unsupported method: '{method}'.")

                records.append({
                    "layer_idx": layer,
                    "x_feat":    x_f,
                    "y_feat":    y_f,
                    "rho":       rho,
                    "pval":      pval,
                    "n_pairs":   len(sub),
                })

    df_records = pd.DataFrame(records)

    # ── Pivot: rows=(x_feat, y_feat), cols=layer_idx ─────────────────────────
    table = (df_records
             .pivot_table(index=["x_feat", "y_feat"],
                          columns="layer_idx",
                          values="rho",
                          aggfunc="first")
             .rename_axis("layer", axis=1))

    # Canonical row order (mirrors input feature list order)
    row_order = [(xf, yf)
                 for xf in x_valid for yf in y_valid
                 if (xf, yf) in table.index]
    table = table.loc[row_order]

    return table, df_records


# =============================================================================
# SYSTEMATIC RUN — all models × all dataset splits
# =============================================================================

def run_layerwise_corr(
    df_combined: pd.DataFrame,
    model_keys: list[str] | None = None,
    x_feats: list[str] | None = None,
    y_feats: list[str] | None = None,
    method: str = "spearman",
    splits: list[str] | None = None,
) -> dict[str, dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    """
    Runs compute_layerwise_corr over the full experiment grid:
        M (models) × D (dataset splits)

    Parameters
    ----------
    df_combined : Combined DataFrame with "model_name" column.
    model_keys  : Subset of MODEL_CONFIGS keys. Defaults to all.
    x_feats     : X feature override. Defaults to DEFAULT_LAYERWISE_X_FEATS.
    y_feats     : Y feature override. Defaults to DEFAULT_LAYERWISE_Y_FEATS.
    method      : "spearman" or "pearson".
    splits      : Dataset splits to compute. Defaults to ["all","wikitext","fineweb"].

    Returns
    -------
    Nested dict: results[model_key][split] = (table, records)
    """
    if model_keys is None:
        model_keys = list(MODEL_CONFIGS.keys())
    if splits is None:
        splits = ["all", "wikitext", "fineweb"]

    results = {}

    for model_key in model_keys:
        cfg      = MODEL_CONFIGS[model_key]
        df_model = df_combined[df_combined["model_name"] == cfg["model_name"]].copy()

        if df_model.empty:
            print(f"[WARNING] No data for '{model_key}'. Skipping.")
            continue

        results[model_key] = {}
        print(f"\n{'═' * 60}")
        print(f"  {cfg['label']}  |  method={method}")
        print(f"{'═' * 60}")

        for split in splits:
            print(f"  › {split:<10}", end="  ")
            try:
                table, records = compute_layerwise_corr(
                    df_model, cfg,
                    x_feats=x_feats,
                    y_feats=y_feats,
                    method=method,
                    dataset=split,
                )
                results[model_key][split] = (table, records)
                max_abs = records["rho"].abs().max()
                n_sig   = (records["pval"] < 0.05).sum()
                print(f"max|ρ|={max_abs:.3f}  |  "
                      f"significant pairs (p<0.05): {n_sig}/{len(records)}")
            except ValueError as e:
                print(f"[SKIP] {e}")

    return results


# =============================================================================
# DISPLAY HELPER — formatted pivot table
# =============================================================================

def display_layerwise_table(
    results: dict,
    model_key: str,
    split: str = "all",
) -> None:
    """
    Pretty-prints the pivot table for a given (model, split) combination.
    Values are ρ coefficients coloured by sign and magnitude.
    """
    from IPython.display import display as ipy_display

    if model_key not in results or split not in results[model_key]:
        print(f"[ERROR] No results for ('{model_key}', '{split}').")
        return

    table, _ = results[model_key][split]
    cfg_label = MODEL_CONFIGS[model_key]["label"]

    print(f"\n{'═' * 60}")
    print(f"  {cfg_label}  |  dataset={split}  |  rows=(x,y)  cols=layer")
    print(f"{'═' * 60}")

    with pd.option_context(
        "display.float_format", "{:+.2f}".format,
        "display.max_columns", None,
        "display.width", 240,
    ):
        ipy_display(
            table.style.background_gradient(
                cmap="RdBu_r", vmin=-1, vmax=1, axis=None
            ).format("{:+.2f}", na_rep="—")
        )


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, pearsonr, theilslopes
from IPython.display import display


DEFAULT_SINGLE_LAYER_X = [
    "rope_pair_var_Wq", "rope_pair_var_Wk",
    "rope_pair_max_ratio_Wq", "rope_pair_max_ratio_Wk",
    "rope_freq_com_Wq", "rope_freq_com_Wk",
]

DEFAULT_SINGLE_LAYER_Y = [
    "diagonal_mass_1", "diagonal_mass_5",
    "diagonal_mass_1_shifted_1", "diagonal_mass_1_shifted_2",
    "diagonal_mass_1_shifted_3", "diagonal_mass_1_shifted_4",
]


def _filter_dataset(df: pd.DataFrame, model_cfg: dict, dataset: str) -> pd.DataFrame:
    if dataset == "wikitext":
        return df[df["prompt_source"] == model_cfg["ptype_wiki"]].copy()
    if dataset == "fineweb":
        return df[df["prompt_source"] == model_cfg["ptype_fineweb"]].copy()
    if dataset == "all":
        return df.copy()
    raise ValueError("dataset must be 'all', 'wikitext', or 'fineweb'")


def _build_single_layer_plot_df(
    df_layer: pd.DataFrame,
    x_feats: list[str],
    y_feats: list[str],
    model_cfg: dict,
    agg: str | None,
) -> pd.DataFrame:
    """
    Costruisce il dataframe da plottare al layer richiesto.

    agg=None      -> una riga per (layer, head, prompt)
    agg='median'  -> una riga per (layer, head), target/input dep aggregati per mediana
    agg='mean'    -> una riga per (layer, head), target/input dep aggregati per media
    """
    all_feats = list(dict.fromkeys(x_feats + y_feats))
    df_A, df_B = _build_corr_df(df_layer, all_feats, model_cfg)

    if df_B is None:
        plot_df = df_A.reset_index()
    else:
        plot_df = df_B.reset_index()

    if agg is not None and df_B is not None:
        agg_fn = "median" if agg == "median" else "mean"
        group_cols = ["layer_idx", "head_idx"]

        numeric_cols = [
            c for c in plot_df.columns
            if c not in group_cols + (["prompt_idx"] if "prompt_idx" in plot_df.columns else [])
            and pd.api.types.is_numeric_dtype(plot_df[c])
        ]

        plot_df = (
            plot_df.groupby(group_cols)[numeric_cols]
            .agg(agg_fn)
            .reset_index()
        )

    model_dep_feats = [f for f in x_feats if f in df_A.columns]
    if model_dep_feats:
        plot_df = plot_df.merge(
            df_A[model_dep_feats].reset_index(),
            on=["layer_idx", "head_idx"],
            how="left",
            suffixes=("", "_md"),
        )
        for f in model_dep_feats:
            if f"{f}_md" in plot_df.columns:
                plot_df[f] = plot_df[f"{f}_md"]
                plot_df = plot_df.drop(columns=[f"{f}_md"])

    return plot_df


def _corr_with_pvalue(sub: pd.DataFrame, x_f: str, y_f: str, method: str):
    if len(sub) < 5:
        return np.nan, np.nan

    if method == "spearman":
        rho, pval = spearmanr(sub[x_f].values, sub[y_f].values)
        return rho, pval

    if method == "pearson":
        rho, pval = pearsonr(sub[x_f].values, sub[y_f].values)
        return rho, pval

    raise ValueError("method must be 'spearman' or 'pearson'")


def _short_label(name: str) -> str:
    return (
        name.replace("diagonal_mass", "diag")
            .replace("shifted", "sh")
            .replace("effective_rank", "erank")
            .replace("rope_pair_max_ratio", "rope max")
            .replace("rope_pair_var", "rope var")
            .replace("rope_freq_com", "rope com")
            .replace("_", " ")
    )


def plot_scatter_single_layer(
    df_combined: pd.DataFrame,
    layer_idx: int,
    x_feats: list[str] | None = None,
    y_feats: list[str] | None = None,
    method: str = "spearman",
    dataset: str = "all",
    agg: str | None = "median",
    model_keys: list[str] | None = None,
    save: bool = SAVE_PLOTS,
):
    """
    Scatterplot pairwise per singolo layer.

    Parametri
    ---------
    df_combined : dataframe combinato con colonna 'model_name'
    layer_idx   : layer da analizzare
    x_feats     : feature model-dependent per asse X
    y_feats     : feature target per asse Y
    method      : 'spearman' o 'pearson'
    dataset     : 'all' | 'wikitext' | 'fineweb'
    agg         : None | 'median' | 'mean'
                  None     -> punti raw (head, prompt)
                  median   -> un punto per head, mediana sui prompt
                  mean     -> un punto per head, media sui prompt
    model_keys  : lista chiavi MODEL_CONFIGS; default = tutti i modelli
    save        : salva PNG in out_dir/<model_name>/
    """
    assert agg in (None, "median", "mean")
    assert dataset in ("all", "wikitext", "fineweb")

    if model_keys is None:
        model_keys = list(MODEL_CONFIGS.keys())

    x_feats = x_feats or DEFAULT_SINGLE_LAYER_X
    y_feats = y_feats or DEFAULT_SINGLE_LAYER_Y

    for model_key in model_keys:
        cfg = MODEL_CONFIGS[model_key]
        df_model = df_combined[df_combined["model_name"] == cfg["model_name"]].copy()

        if df_model.empty:
            print(f"[SKIP] {cfg['label']}: dataframe vuoto")
            continue

        df_model = _filter_dataset(df_model, cfg, dataset)
        df_layer = df_model[df_model["layer_idx"] == layer_idx].copy()

        if df_layer.empty:
            print(f"[SKIP] {cfg['label']}: nessun dato per layer={layer_idx}, dataset={dataset}")
            continue

        valid_x = [f for f in x_feats if f in df_layer.columns]
        valid_y = [f for f in y_feats if f in df_layer.columns]

        if not valid_x or not valid_y:
            print(f"[SKIP] {cfg['label']}: nessuna feature valida")
            continue

        plot_df = _build_single_layer_plot_df(df_layer, valid_x, valid_y, cfg, agg)

        heads = sorted(plot_df["head_idx"].dropna().unique())
        if len(heads) == 0:
            print(f"[SKIP] {cfg['label']}: nessuna head disponibile")
            continue

        n_x, n_y = len(valid_x), len(valid_y)

        cmap = cm.get_cmap("viridis", len(heads))
        norm = mcolors.Normalize(vmin=min(heads), vmax=max(heads))

        cell_w = 3.1
        cell_h = 2.8
        fig_w = 1.0 + n_x * cell_w + 0.9
        fig_h = 1.0 + n_y * cell_h + 0.8

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=170)
        gs = gridspec.GridSpec(
            n_y, n_x,
            left=0.08, right=0.88, top=0.88, bottom=0.12,
            hspace=0.22, wspace=0.16
        )
        axes = np.array([
            [fig.add_subplot(gs[r, c]) for c in range(n_x)]
            for r in range(n_y)
        ])

        point_size  = 52 if agg is not None else 14
        point_alpha = 0.90 if agg is not None else 0.60
        edge_width  = 0.70 if agg is not None else 0.00

        for row, y_f in enumerate(valid_y):
            for col, x_f in enumerate(valid_x):
                ax = axes[row, col]

                sub = plot_df[[x_f, y_f, "head_idx"]].dropna()
                if sub.empty:
                    ax.set_visible(False)
                    continue

                ax.scatter(
                    sub[x_f], sub[y_f],
                    c=sub["head_idx"], cmap=cmap, norm=norm,
                    s=point_size, alpha=point_alpha,
                    edgecolors="#2f2f2f" if agg is not None else "none",
                    linewidths=edge_width,
                    zorder=2,
                    rasterized=True,
                )

                if len(sub) > 5 and sub[x_f].nunique() > 1 and sub[y_f].nunique() > 1:
                    try:
                        res = theilslopes(sub[y_f].values, sub[x_f].values)
                        x_fit = np.array([sub[x_f].min(), sub[x_f].max()])
                        ax.plot(
                            x_fit, res.slope * x_fit + res.intercept,
                            color="#c62828", lw=1.2, ls="--", zorder=4, alpha=0.9
                        )
                    except Exception:
                        pass

                    rho, pval = _corr_with_pvalue(sub, x_f, y_f, method)
                    is_sig = pd.notna(pval) and pval < 0.05
                    ann_color = "#c62828" if pd.notna(rho) and abs(rho) > 0.4 and is_sig else "#4d4d4d"
                    star = "*" if is_sig else ""

                    ax.annotate(
                        f"ρ={rho:.2f}{star}" if pd.notna(rho) else "ρ=nan",
                        xy=(0.04, 0.90), xycoords="axes fraction",
                        fontsize=7, color=ann_color,
                        fontweight="bold" if is_sig else "normal",
                        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.82),
                        zorder=5,
                    )

                if row == n_y - 1:
                    ax.set_xlabel(_short_label(x_f), fontsize=7, labelpad=3)
                else:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")

                if col == 0:
                    ax.set_ylabel(_short_label(y_f), fontsize=7, labelpad=3)
                else:
                    ax.set_yticklabels([])
                    ax.set_ylabel("")

                ax.tick_params(labelsize=6, length=3, width=0.5)
                ax.grid(True, lw=0.3, alpha=0.30, color="#bdbdbd", zorder=1)

                for spine in ax.spines.values():
                    spine.set_linewidth(0.6)
                    spine.set_edgecolor("#b8b8b8")

        cax = fig.add_axes([0.905, 0.16, 0.015, 0.62])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Head index", fontsize=8, labelpad=6)
        cbar.ax.tick_params(labelsize=7)

        agg_label = "raw" if agg is None else agg
        fig.text(
            0.48, 0.965,
            "Model-dependent (X) × Target (Y)",
            ha="center", va="top",
            fontsize=11, fontweight="bold"
        )
        fig.text(
            0.48, 0.938,
            f"{cfg['label']} | layer {layer_idx} | {dataset} | agg: {agg_label} | method: {method}",
            ha="center", va="top",
            fontsize=9, color="#333333"
        )

        legend_handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#777777", markeredgecolor="#2f2f2f",
                   markersize=6 if agg is not None else 4,
                   label="points"),
            Line2D([0], [0], color="#c62828", lw=1.2, ls="--",
                   label="Theil-Sen fit"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=2,
            frameon=True,
            framealpha=0.95,
            edgecolor="#d0d0d0",
            fontsize=8,
            bbox_to_anchor=(0.48, 0.03),
            bbox_transform=fig.transFigure,
        )

        if save:
            model_dir = out_dir / cfg["model_name"]
            model_dir.mkdir(exist_ok=True)
            fname = f"scatter_layer_{layer_idx}_{dataset}_{agg_label}_{method}.png"
            fig.savefig(model_dir / fname, dpi=220, bbox_inches="tight")

        display(fig)
        plt.close(fig)


# =============================================================================
# INTERACTIVE 3D SCATTER — Q-Sim × K-Sim × Target
# Plotly scatter_3d: each point = (layer, head, prompt).
# Multi-model support via MODEL_CONFIGS + df_combined.
# =============================================================================

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# DEFAULT AXIS OPTIONS
# (Override at call time via x_feat / y_feat / z_feat parameters)
# =============================================================================

DEFAULT_3D_X = "q_sim_consecutive"
DEFAULT_3D_Y = "k_sim_consecutive"
DEFAULT_3D_Z = "diagonal_mass_1"

# All recognised Z choices — used to validate input
VALID_Z_FEATS = [
    "diagonal_mass_1", "diagonal_mass_5",
    "diagonal_mass_1_shifted_1", "diagonal_mass_1_shifted_2",
    "diagonal_mass_1_shifted_3", "diagonal_mass_1_shifted_4",
    "sink_mass_token_0", "sink_mass_token_1",
    "sink_mass_max", "look_back", "attention_gini",
    "effective_rank_A", "r95_A",
]


# =============================================================================
# STYLING HELPERS
# =============================================================================

_AXIS_STYLE = dict(
    backgroundcolor="#f5f5f5",
    gridcolor="#dddddd",
    showbackground=True,
    zerolinecolor="#cccccc",
    tickfont=dict(size=10),
)

def _short(name: str) -> str:
    return (
        name.replace("diagonal_mass", "diag")
            .replace("shifted", "sh")
            .replace("effective_rank", "erank")
            .replace("_consecutive", "")
            .replace("_", " ")
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

def plot_interactive_3d(
    df_combined: pd.DataFrame,
    model_keys: list[str] | None = None,
    x_feat: str = DEFAULT_3D_X,
    y_feat: str = DEFAULT_3D_Y,
    z_feat: str = DEFAULT_3D_Z,
    color_by: str = "layer_idx",
    dataset: str = "all",
    max_points: int | None = 30_000,
    save_html: bool = True,
) -> None:
    """
    Interactive Plotly 3D scatter: x_feat × y_feat × z_feat.
    One figure per model.

    Parameters
    ----------
    df_combined : Combined DataFrame with "model_name" column.
    model_keys  : Subset of MODEL_CONFIGS. Defaults to all.
    x_feat      : Feature for X axis. Default: q_sim_consecutive.
    y_feat      : Feature for Y axis. Default: k_sim_consecutive.
    z_feat      : Feature for Z axis. Default: diagonal_mass_1.
    color_by    : Column used for colour coding.
                  "layer_idx" | "head_idx" | "prompt_source"
    dataset     : "all" | "wikitext" | "fineweb"
    max_points  : Cap on plotted rows (random subsample) to keep browser fast.
                  Set None to disable.
    save_html   : Saves interactive HTML to out_dir/<model_name>/.
    """
    assert color_by in ("layer_idx", "head_idx", "prompt_source"), \
        "color_by must be 'layer_idx', 'head_idx', or 'prompt_source'."
    assert dataset in ("all", "wikitext", "fineweb"), \
        "dataset must be 'all', 'wikitext', or 'fineweb'."

    if model_keys is None:
        model_keys = list(MODEL_CONFIGS.keys())

    for model_key in model_keys:
        cfg      = MODEL_CONFIGS[model_key]
        df_model = df_combined[df_combined["model_name"] == cfg["model_name"]].copy()

        if df_model.empty:
            print(f"[SKIP] {cfg['label']}: dataframe vuoto")
            continue

        # ── Dataset filter ────────────────────────────────────────────────────
        if dataset == "wikitext":
            df_f = df_model[df_model["prompt_source"] == cfg["ptype_wiki"]]
        elif dataset == "fineweb":
            df_f = df_model[df_model["prompt_source"] == cfg["ptype_fineweb"]]
        else:
            df_f = df_model

        # ── Feature validation ────────────────────────────────────────────────
        required = [x_feat, y_feat, z_feat]
        missing  = [f for f in required if f not in df_f.columns]
        if missing:
            print(f"[SKIP] {cfg['label']}: missing features — {missing}")
            continue

        # ── Hover columns ─────────────────────────────────────────────────────
        hover_cols = ["layer_idx", "head_idx"]
        for opt in ["prompt_idx", "prompt_source"]:
            if opt in df_f.columns and opt not in hover_cols:
                hover_cols.append(opt)
                break

        # ── Drop NaN, optional subsample ─────────────────────────────────────
        df_plot = df_f[list(dict.fromkeys(required + hover_cols))].dropna(
            subset=required
        ).copy()

        if max_points is not None and len(df_plot) > max_points:
            df_plot = df_plot.sample(n=max_points, random_state=42)
            sampled = True
        else:
            sampled = False

        n_pts = len(df_plot)
        print(f"\n── {cfg['label']}  |  dataset={dataset}  |  "
              f"n={n_pts:,}{'  (sampled)' if sampled else ''}")

        # ── Colour scale ──────────────────────────────────────────────────────
        use_discrete = (color_by == "prompt_source")
        color_scale  = "Plasma" if not use_discrete else None

        # ── Figure ───────────────────────────────────────────────────────────
        subtitle = (
            f"{cfg['label']} | {dataset}"
            f"{' (sampled '+str(max_points//1000)+'k)' if sampled else ''}"
            f" | colour: {color_by}"
        )

        if use_discrete:
            fig = px.scatter_3d(
                df_plot,
                x=x_feat, y=y_feat, z=z_feat,
                color=color_by,
                hover_data=hover_cols,
                opacity=0.60,
                title=(
                    f"<b>{_short(x_feat)} × {_short(y_feat)} × {_short(z_feat)}</b>"
                    f"<br><sup>{subtitle}</sup>"
                ),
            )
        else:
            fig = px.scatter_3d(
                df_plot,
                x=x_feat, y=y_feat, z=z_feat,
                color=color_by,
                color_continuous_scale=color_scale,
                hover_data=hover_cols,
                opacity=0.55,
                title=(
                    f"<b>{_short(x_feat)} × {_short(y_feat)} × {_short(z_feat)}</b>"
                    f"<br><sup>{subtitle}</sup>"
                ),
            )

        # ── Marker style ──────────────────────────────────────────────────────
        fig.update_traces(
            marker=dict(
                size=2.5 if n_pts > 5_000 else 4.0,
                line=dict(width=0),
            )
        )

        # ── Axis labels ───────────────────────────────────────────────────────
        fig.update_layout(
            scene=dict(
                xaxis_title=_short(x_feat),
                yaxis_title=_short(y_feat),
                zaxis_title=_short(z_feat),
                xaxis={**_AXIS_STYLE},
                yaxis={**_AXIS_STYLE},
                zaxis={**_AXIS_STYLE},
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=70),
            paper_bgcolor="white",
            font=dict(family="DejaVu Sans, Arial, sans-serif", size=11),
            coloraxis_colorbar=dict(
                title=dict(text=color_by.replace("_", " "), side="right"),
                thickness=14,
                len=0.65,
            ),
        )

        # ── Save ──────────────────────────────────────────────────────────────
        if save_html:
            model_dir = out_dir / cfg["model_name"]
            model_dir.mkdir(parents=True, exist_ok=True)
            fname = f"3d_{_short(z_feat).replace(' ', '_')}_{dataset}_{color_by}.html"
            fig.write_html(str(model_dir / fname))
            print(f"  saved → {model_dir / fname}")

        fig.show()

