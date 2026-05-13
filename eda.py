"""EDA functions extracted from data_analysis.ipynb."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from IPython.display import display

from scipy.stats import spearmanr

# Palette e stile + constants
C_WIKI  = "#4878CF"
C_RAND  = "#D65F5F"
C_MODEL = "#2CA02C"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#CCCCCC", "grid.linewidth": 0.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlepad": 6,
    "font.family": "DejaVu Sans",
})

PTYPE_WIKI      = "wikitext_wikitext-103-raw-v1_train"
PTYPE_RAND      = "random_vocab"
N_COLS          = 2
STRIP_THRESHOLD = 80
RANK_PADDING    = 5
SAVE_PLOTS      = False
out_dir         = Path("output"); out_dir.mkdir(exist_ok=True)

TAXONOMY = {

    "model_dependent": {

        "Weight Matrix Ranks Wq": {
            "features":    ["effective_rank_Wq", "r95_Wq"],
            "bounds":      {},
            "rank_max":    128,
            "use_logy":    False, "use_density": True,
            "show_split":  False,
            "notes": "Per-head slice of W_q. 32 distinct heads in Mistral-7B.",
        },

        "Weight Matrix Ranks Wk Wv": {
            "features":    ["effective_rank_Wk","r95_Wk","effective_rank_Wv","r95_Wv"],
            "bounds":      {},
            "rank_max":    128,
            "use_logy":    False, "use_density": True,
            "show_split":  False,
            "notes": "Per-KV-head slice of W_k/W_v. 8 distinct KV-heads in Mistral-7B (GQA).",
        },

        "Gini Spectral Concentration": {
            "features":    ["gini_left_Wq","gini_right_Wq","gini_left_Wk","gini_right_Wk"],
            "bounds":      {f:(0.0,1.0) for f in
                            ["gini_left_Wq","gini_right_Wq","gini_left_Wk","gini_right_Wk"]},
            "use_logy":    False, "use_density": True,
            "show_split":  False,
            "notes": "Gini over left/right singular vectors of W_q/W_k. In [0,1].",
        },

        "RMSNorm": {
            "features":    ["rmsnorm_gamma_norm"],
            "bounds":      {"rmsnorm_gamma_norm":(0.0, None)},
            "use_logy":    False, "use_density": True,
            "show_split":  False,
            "notes": "L2 norm of RMSNorm gamma. Fixed per layer.",
        },

        "RoPE Structure": {
            "features":    ["rope_pair_var_Wq","rope_pair_var_Wk",
                            "rope_pair_max_ratio_Wq","rope_pair_max_ratio_Wk",
                            "rope_freq_com_Wq","rope_freq_com_Wk"],
            "bounds":      {"rope_pair_var_Wq":(0,None),"rope_pair_var_Wk":(0,None),
                            "rope_pair_max_ratio_Wq":(0,None),"rope_pair_max_ratio_Wk":(0,None),
                            "rope_freq_com_Wq":(0,63.0),"rope_freq_com_Wk":(0,63.0)},
            "use_logy":    False, "use_density": True,
            "show_split":  False,
            "notes": "rope_freq_com in [0,63]. rope_pair_max_ratio: dominance of top RoPE pair.",
        },

        "RoPE-aware QK Alignment": {
            "features":    ["compute_WqRWk_alignment_delta_0"],
            "bounds":      {"compute_WqRWk_alignment_delta_0":(0,1)},
            "use_logy":    False, "use_density": True,
            "show_split":  False,
            "notes": "Cosine alignment W_q * R(delta=0) * W_k^T. Pure weight property.",
        },

        
    },

    "input_dependent": {

        "Hidden State Rank": {
            "features":    ["effective_rank_H","r95_H"],
            "bounds":      {},
            "rank_max":    512,
            "use_logy":    False, "use_density": True,
            "show_split":  True,
            "notes": "H shape (seq_len, 4096). rank_max=min(seq_len,4096).",
        },

        "Projected Q/K Ranks": {
            "features":    ["effective_rank_Q","r95_Q","effective_rank_K","r95_K"],
            "bounds":      {},
            "rank_max":    128,
            "use_logy":    False, "use_density": True,
            "show_split":  True,
            "notes": "Q=H*W_q^T, K=H*W_k^T per head. Shape (seq,128).",
        },

        "Temporal Similarity": {
            "features":    ["q_sim_consecutive","k_sim_consecutive"],
            "bounds":      {"q_sim_consecutive":(0,1),"k_sim_consecutive":(0,1)},
            "use_logy":    False, "use_density": True,
            "show_split":  True,
            "notes": "Mean cosine sim between consecutive token rows of Q/K.",
        },
        "SVD Alignment H vs W": {
            "features":    ["svd_alignment_H_Wq","svd_alignment_H_Wk"],
            "bounds":      {"svd_alignment_H_Wq":(0,1),"svd_alignment_H_Wk":(0,1)},
            "use_logy":    False, "use_density": True,
            "show_split":  True,
            "notes": "Cosine sim top singular vector of H vs W_q/W_k. Mixed: W fixed, H input-dep.",
        },
    },

    "target": {

        "Attention Map Diagonal Mass": {
            "features":    ["diagonal_mass_1","diagonal_mass_5",
                            "diagonal_mass_1_shifted_1","diagonal_mass_1_shifted_2",
                            "diagonal_mass_1_shifted_3","diagonal_mass_1_shifted_4"],
            "bounds":      {f:(0,1) for f in
                            ["diagonal_mass_1","diagonal_mass_5","diagonal_mass_1_shifted_1",
                             "diagonal_mass_1_shifted_2","diagonal_mass_1_shifted_3",
                             "diagonal_mass_1_shifted_4"]},
            "use_logy":    True, "use_density": True,
            "show_split":  True,
            "notes": "Attention mass on main diagonal (width 1 or 5) and sub-diagonals.",
        },

        "Attention Map Sink Mass": {
            "features":    ["sink_mass_token_0","sink_mass_token_1","sink_mass_token_2",
                            "sink_mass_token_3","sink_mass_token_4","sink_mass_max"],
            "bounds":      {f:(0,1) for f in
                            ["sink_mass_token_0","sink_mass_token_1","sink_mass_token_2",
                             "sink_mass_token_3","sink_mass_token_4","sink_mass_max"]},
            "use_logy":    True, "use_density": True,
            "show_split":  True,
            "notes": "Attention mass on first 5 tokens (sinks). sink_mass_max = row-wise max.",
        },

        "Attention Map Structure": {
            "features":    ["look_back","attention_gini"],
            "bounds":      {"look_back":(0,1),"attention_gini":(0,1)},
            "use_logy":    False, "use_density": True,
            "show_split":  True,
            "notes": "look_back: mean backward distance norm. attention_gini: row Gini.",
        },

        "Attention Matrix Rank": {
            "features":    ["effective_rank_A","r95_A"],
            "bounds":      {},
            "rank_max":    512,
            "use_logy":    False, "use_density": True,
            "show_split":  True,
            "notes": "Rank of post-softmax A. Low=sparse, High=diffuse.",
        },
    },
}

FEATURE_DEDUP = {
    "effective_rank_Wq":               ["layer_idx", "head_idx"],
    "r95_Wq":                          ["layer_idx", "head_idx"],
    "gini_left_Wq":                    ["layer_idx", "head_idx"],
    "gini_right_Wq":                   ["layer_idx", "head_idx"],
    "rope_pair_var_Wq":                ["layer_idx", "head_idx"],
    "rope_pair_max_ratio_Wq":          ["layer_idx", "head_idx"],
    "rope_freq_com_Wq":                ["layer_idx", "head_idx"],
    "compute_WqRWk_alignment_delta_0": ["layer_idx", "head_idx"],

    "effective_rank_Wk":               ["layer_idx", "kv_head"],
    "r95_Wk":                          ["layer_idx", "kv_head"],
    "effective_rank_Wv":               ["layer_idx", "kv_head"],
    "r95_Wv":                          ["layer_idx", "kv_head"],
    "gini_left_Wk":                    ["layer_idx", "kv_head"],
    "gini_right_Wk":                   ["layer_idx", "kv_head"],
    "rope_pair_var_Wk":                ["layer_idx", "kv_head"],
    "rope_pair_max_ratio_Wk":          ["layer_idx", "kv_head"],
    "rope_freq_com_Wk":                ["layer_idx", "kv_head"],

    "rmsnorm_gamma_norm":              ["layer_idx"],
}

def _prepare_vals(df, feat, show_split):
    dedup_cols = FEATURE_DEDUP.get(feat, None)

    if dedup_cols is not None:
        if "kv_head" in dedup_cols and "kv_head" not in df.columns:
            df = df.copy()
            df["kv_head"] = df["head_idx"] // 4
        subset = [c for c in dedup_cols if c in df.columns]
        sub = df.drop_duplicates(subset=subset) if subset else df
    else:
        sub = df

    if show_split:
        return {
            "wikitext": sub[sub["prompt_source"] == PTYPE_WIKI][feat].dropna().values,
            "random":   sub[sub["prompt_source"] == PTYPE_RAND][feat].dropna().values,
        }
    else:
        return {"all": sub[feat].dropna().values}

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

def _compute_rank_bounds(vals_list, rank_max, padding=RANK_PADDING):
    all_v = np.concatenate([v for v in vals_list if len(v)])
    if len(all_v) == 0:
        return 1, (rank_max if rank_max is not None else 1)

    lo = max(1, int(np.floor(all_v.min())) - padding)
    hi = int(np.ceil(all_v.max())) + padding
    if rank_max is not None:
        hi = min(hi, rank_max)
    return lo, hi

def _build_feat_bounds(valid, meta, df):
    bounds     = meta["bounds"]
    show_split = meta["show_split"]
    feat_bounds = {}

    if "rank_max" in meta:
        rank_max      = meta["rank_max"]
        suffix_groups = _group_rank_features_by_suffix(valid)
        for suffix, suffix_feats in suffix_groups.items():
            group_vals = []
            for f in suffix_feats:
                sv = _prepare_vals(df, f, show_split)
                group_vals.extend(v for v in sv.values() if len(v))
            lo_p, hi_p = _compute_rank_bounds(group_vals, rank_max)
            for f in suffix_feats:
                feat_bounds[f] = (lo_p, hi_p)
    else:
        for f in valid:
            sv    = _prepare_vals(df, f, show_split)
            all_v = np.concatenate(list(sv.values()))

            if len(all_v) == 0:
                feat_bounds[f] = (0.0, 1.0)
                continue

            lo, hi = bounds.get(f, (None, None))
            lo_p   = lo if lo is not None else float(all_v.min()) - abs(float(all_v.min())) * 0.02
            hi_p   = hi if hi is not None else float(all_v.max()) * 1.02
            feat_bounds[f] = (lo_p, hi_p)

    return feat_bounds

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

def _choose_plot_type(df, meta):
    for feat in meta["features"]:
        if feat in df.columns:
            n = sum(len(v) for v in _prepare_vals(df, feat, meta["show_split"]).values())
            return "strip" if n < STRIP_THRESHOLD else "hist"
    return "hist"

QUARTILE_COLORS = ["#3B4CC0", "#6EC6C6", "#F4A44A", "#D65F5F"]
QUARTILE_LABELS = ["Q1 (early)", "Q2", "Q3", "Q4 (late)"]

def _plot_hist(group_name, meta, df):
    from IPython.display import display

    features    = meta["features"]
    use_logy    = meta["use_logy"]
    use_density = meta["use_density"]
    show_split  = meta["show_split"]

    valid   = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  [SKIP] {group_name}: {missing}")
    if not valid:
        return

    feat_bounds = _build_feat_bounds(valid, meta, df)

    use_layer_quartiles = (
        not show_split
        and "layer_idx" in df.columns
        and df["layer_idx"].nunique() >= 4
    )

    if use_layer_quartiles:
        layers     = sorted(df["layer_idx"].unique())
        n_layers   = len(layers)
        q_size     = n_layers // 4
        layer_to_q = {}
        for i, l in enumerate(layers):
            q = min(i // q_size, 3)
            layer_to_q[l] = q

    n_cols = min(N_COLS, len(valid))
    n_rows = int(np.ceil(len(valid) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5*n_cols, 3.6*n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(n_rows, n_cols)

    palette      = {"wikitext": C_WIKI, "random": C_RAND, "all": C_MODEL}
    labels       = {"wikitext": "Wikitext", "random": "Random", "all": "all heads"}
    legend_drawn = False

    for idx, feat in enumerate(valid):
        row, col = divmod(idx, n_cols)
        ax       = axes[row, col]

        lo_p, hi_p = feat_bounds[feat]

        if use_layer_quartiles:
            dedup = FEATURE_DEDUP.get(feat, None)
            tmp   = df.copy()
            if dedup is not None:
                if "kv_head" in dedup and "kv_head" not in tmp.columns:
                    tmp["kv_head"] = tmp["head_idx"] // 4
                key_cols = [c for c in dedup if c in tmp.columns]
                tmp = tmp.drop_duplicates(subset=key_cols)

            tmp = tmp[["layer_idx", feat]].dropna()
            if tmp.empty:
                ax.set_visible(False)
                continue

            tmp["quartile"] = tmp["layer_idx"].map(layer_to_q)
            q_vals = [
                np.clip(tmp.loc[tmp["quartile"] == q, feat].values, lo_p, hi_p)
                for q in range(4)
            ]
            q_vals = [v for v in q_vals if len(v) > 0]

            all_v = np.concatenate(q_vals)
            if len(all_v) == 0:
                ax.set_visible(False)
                continue

            bins = _compute_bins(q_vals, lo_p, hi_p,
                                 min_bins=meta.get("min_bins", 10),
                                 max_bins=meta.get("max_bins", 150))

            ax.hist(q_vals, bins=bins,
                    stacked=True,
                    color=QUARTILE_COLORS[:len(q_vals)],
                    label=QUARTILE_LABELS[:len(q_vals)],
                    alpha=0.85,
                    density=use_density,
                    edgecolor="none")

            if not legend_drawn:
                ax.legend(title="Layer depth",
                          title_fontsize=7, loc="upper right", fontsize=7)
                legend_drawn = True

        else:
            split_vals = _prepare_vals(df, feat, show_split)
            all_v      = np.concatenate(list(split_vals.values()))

            if len(all_v) == 0:
                ax.set_visible(False)
                continue

            bins = _compute_bins(list(split_vals.values()), lo_p, hi_p,
                                 min_bins=meta.get("min_bins", 10),
                                 max_bins=meta.get("max_bins", 150))

            for key, vals in split_vals.items():
                vc = np.clip(vals, lo_p, hi_p)
                ax.hist(vc, bins=bins, color=palette[key], alpha=0.65,
                        label=labels[key], density=use_density)
                if len(vc):
                    ax.axvline(np.mean(vc),   color=palette[key], lw=1.4, ls="-")
                    ax.axvline(np.median(vc), color=palette[key], lw=1.4, ls="--")

            if not legend_drawn:
                ax.legend(title="— mean   -- median",
                          title_fontsize=7, loc="upper right", fontsize=7)
                legend_drawn = True

        ax.set_xlim(lo_p, hi_p)
        if use_logy:
            ax.set_yscale("log")
        ax.set_ylabel(("density" if use_density else "count") if col == 0 else "")
        ax.set_title(feat.replace("_", " "), pad=4)

    for idx in range(len(valid), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    tags      = ("  [log y]" if use_logy else "") + ("  [density]" if use_density else "  [count]")
    split_tag = "  [layer quartiles]" if use_layer_quartiles else \
                ("  [Wikitext vs Random]" if show_split else "  [model only]")
    fig.suptitle(f"{group_name}{tags}{split_tag}  [hist]",
                 fontsize=10, fontweight="bold", y=1.05)

    if SAVE_PLOTS:
        _save_fig(fig, f"hist_{group_name}")
    display(fig)
    plt.close(fig)

def _plot_strip(group_name, meta, df):
    from IPython.display import display

    features   = meta["features"]
    use_logy   = meta["use_logy"]
    show_split = meta["show_split"]

    valid   = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  [SKIP] {group_name}: {missing}")
    if not valid:
        return

    feat_bounds = _build_feat_bounds(valid, meta, df)

    n_cols = min(N_COLS, len(valid))
    n_rows = int(np.ceil(len(valid) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5*n_cols, 3.2*n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(n_rows, n_cols)

    palette = {"wikitext": C_WIKI, "random": C_RAND, "all": C_MODEL}
    labels  = {"wikitext": "Wikitext", "random": "Random", "all": "all heads"}
    rng     = np.random.default_rng(42)

    for idx, feat in enumerate(valid):
        row, col   = divmod(idx, n_cols)
        ax         = axes[row, col]
        split_vals = _prepare_vals(df, feat, show_split)
        all_v      = np.concatenate(list(split_vals.values()))

        if len(all_v) == 0:
            ax.set_visible(False)
            continue

        lo_p, hi_p = feat_bounds[feat]
        box_data, box_pos, box_colors = [], [], []

        for i, (key, vals) in enumerate(split_vals.items()):
            if len(vals) == 0:
                continue
            color  = palette[key]
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, alpha=0.7, s=22, linewidths=0,
                       label=labels[key], zorder=3)
            ax.hlines(np.mean(vals),   i - 0.3, i + 0.3,
                      color=color, lw=2.0, ls="-",  zorder=4)
            ax.hlines(np.median(vals), i - 0.3, i + 0.3,
                      color=color, lw=2.0, ls="--", zorder=4)
            box_data.append(vals)
            box_pos.append(i)
            box_colors.append(color)

        if box_data:
            bp = ax.boxplot(box_data, positions=box_pos, widths=0.35,
                            patch_artist=True, showfliers=False, zorder=2,
                            medianprops=dict(linewidth=0),
                            whiskerprops=dict(linewidth=0.8, color="#888"),
                            capprops=dict(linewidth=0.8, color="#888"),
                            boxprops=dict(linewidth=0.8))
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.15)

        keys = list(split_vals.keys())
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([labels[k] for k in keys], fontsize=8)
        ax.set_ylim(lo_p, hi_p)
        if use_logy:
            ax.set_yscale("log")
        ax.set_ylabel(feat.replace("_", " ") if col == 0 else "")
        ax.set_title(feat.replace("_", " "), pad=4)

        if idx == 0:
            handles = [
                Line2D([0],[0], color="gray", lw=2, ls="-",  label="mean"),
                Line2D([0],[0], color="gray", lw=2, ls="--", label="median"),
            ]
            ax.legend(handles=handles, fontsize=7, loc="upper right")

    for idx in range(len(valid), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    split_tag = "  [Wikitext vs Random]" if show_split else "  [model only — per head]"
    fig.suptitle(f"{group_name}{split_tag}  [strip+box]",
                 fontsize=10, fontweight="bold", y=1.05)

    if SAVE_PLOTS:
        _save_fig(fig, f"strip_{group_name}")
    display(fig)
    plt.close(fig)

def _save_fig(fig, name):
    safe = (name.lower()
            .replace(" ","_").replace("(","").replace(")","").replace(",","")
            .replace("/","_").replace("—","").replace("&","")
            .replace("__","_").strip("_"))
    fig.savefig(out_dir / f"{safe}.png", dpi=150, bbox_inches="tight")

def plot_group(group_name, meta, df):
    plot_type = _choose_plot_type(df, meta)
    if plot_type == "strip":
        _plot_strip(group_name, meta, df)
    else:
        _plot_hist(group_name, meta, df)

def _get_all_features(taxonomy, df, exclude_subgroups=None):
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

def _feat_to_category(taxonomy):
    mapping = {}
    for cat_name, subgroups in taxonomy.items():
        for meta in subgroups.values():
            for f in meta["features"]:
                mapping[f] = cat_name
    return mapping

def _make_plotly_heatmap(corr, title, feat_order, feat_to_cat, vmin=-1, vmax=1):
    n = len(feat_order)
    z = corr.values.copy().astype(float)
    for i in range(n):
        for j in range(i + 1, n):
            z[i, j] = np.nan

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
                    f"r = {v:.3f}" if not np.isnan(v) else ""
                )

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z,
        x=feat_order,
        y=feat_order,
        zmin=vmin, zmax=vmax,
        colorscale="RdBu_r",
        zmid=0,
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        showscale=True,
        colorbar=dict(title="r", thickness=14, len=0.7),
        xgap=0.5, ygap=0.5,
    ))

    shapes = []
    current_cat = feat_to_cat.get(feat_order[0])
    for i, f in enumerate(feat_order[1:], start=1):
        cat = feat_to_cat.get(f)
        if cat != current_cat:
            shapes.append(dict(
                type="line",
                x0=i - 0.5, x1=i - 0.5,
                y0=i - 0.5, y1=n - 0.5,
                line=dict(color="black", width=1.5),
                layer="above",
            ))
            shapes.append(dict(
                type="line",
                x0=-0.5,    x1=i - 0.5,
                y0=i - 0.5, y1=i - 0.5,
                line=dict(color="black", width=1.5),
                layer="above",
            ))
            current_cat = cat

    annotations = []
    seg_start, seg_cat = 0, feat_to_cat.get(feat_order[0])
    segments = []
    for i, f in enumerate(feat_order[1:], start=1):
        cat = feat_to_cat.get(f)
        if cat != seg_cat:
            segments.append((seg_start, i, seg_cat))
            seg_start, seg_cat = i, cat
    segments.append((seg_start, n, seg_cat))

    for s, e, cat in segments:
        mid = (s + e - 1) / 2
        annotations.append(dict(
            x=mid, y=n + 0.8,
            text=f"<b>{cat}</b>",
            showarrow=False,
            font=dict(size=10, color="#333"),
            xref="x", yref="y",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#222"), x=0.5),
        xaxis=dict(
            tickangle=45, tickfont=dict(size=8),
            side="bottom", constrain="domain",
            scaleanchor="y",
        ),
        yaxis=dict(
            tickfont=dict(size=8),
            autorange="reversed",
            constrain="domain",
        ),
        shapes=shapes,
        annotations=annotations,
        width=820, height=760,
        margin=dict(l=120, r=60, t=80, b=150),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def plot_gini_boxplot_per_head(
    df,
    prompt_source: str | None = None,
    layer: int | list | None = None,
    figsize_per_cell: tuple = (1.6, 2.0),
    save: bool = False,
):
    """
    Boxplot del Gini index dell'attention matrix per ogni head.
    Singolo layer → layout 4 colonne × ceil(n_heads/4) righe.
    Multi-layer  → layout n_layers × n_heads.
    """
    from IPython.display import display

    feat = "attention_gini"

    if prompt_source is not None:
        df = df[df["prompt_source"] == prompt_source].copy()

    if feat not in df.columns:
        print(f"Feature '{feat}' non trovata nel DataFrame.")
        return

    all_layers = sorted(df["layer_idx"].unique())
    if layer is None:
        layers = all_layers
    elif isinstance(layer, int):
        if layer not in all_layers:
            print(f"Layer {layer} non trovato. Disponibili: {all_layers}")
            return
        layers = [layer]
    elif isinstance(layer, list):
        layers = [l for l in layer if l in all_layers]
        if not layers:
            print("Nessuno dei layer specificati trovato.")
            return
    else:
        raise ValueError("'layer' deve essere None, int o list[int]")

    heads = sorted(df["head_idx"].unique())
    n_h   = len(heads)
    n_l   = len(layers)

    single_layer_mode = (n_l == 1)
    if single_layer_mode:
        n_cols = 4
        n_rows = int(np.ceil(n_h / n_cols))
    else:
        n_cols = n_h
        n_rows = n_l

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        constrained_layout=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    medians_global = df.groupby(["layer_idx", "head_idx"])[feat].median().dropna()
    vmin, vmax = medians_global.min(), medians_global.max()
    cmap = plt.get_cmap("RdYlGn")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rng  = np.random.default_rng(42)

    def _draw_cell(ax, layer_idx, head, title_str, ylabel_str=None):
        sub = df[(df["layer_idx"] == layer_idx) & (df["head_idx"] == head)][feat].dropna()
        if sub.empty:
            ax.set_visible(False)
            return

        median_val = sub.median()
        color      = cmap(norm(median_val))

        ax.boxplot(
            sub.values, widths=0.5, patch_artist=True, notch=False, showfliers=False,
            medianprops=dict(color="#222", lw=1.8),
            whiskerprops=dict(color="#666", lw=0.8),
            capprops=dict(color="#666", lw=0.8),
            boxprops=dict(facecolor=color, alpha=0.80, linewidth=0.5),
        )
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        ax.scatter(np.ones(len(sub)) + jitter, sub.values,
                   s=8, alpha=0.6, color=color,
                   edgecolors="#333", linewidths=0.25, zorder=3)

        ax.text(1.38, median_val, f"{median_val:.2f}",
                va="center", ha="left", fontsize=6,
                color="#333", zorder=4)

        ax.set_title(title_str, fontsize=8 if single_layer_mode else 6, pad=3)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=6)
        ax.set_ylim(vmin - 0.02, vmax + 0.05)
        ax.grid(True, axis="y", lw=0.3, alpha=0.4)
        if ylabel_str:
            ax.set_ylabel(ylabel_str, fontsize=6, rotation=0, labelpad=20, va="center")

    if single_layer_mode:
        layer_idx = layers[0]
        for idx, head in enumerate(heads):
            row, col = divmod(idx, n_cols)
            _draw_cell(axes[row, col], layer_idx, head, title_str=f"Head {head}")
        for idx in range(len(heads), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)
    else:
        for row, layer_idx in enumerate(layers):
            for col, head in enumerate(heads):
                ylabel = f"L{layer_idx}" if col == 0 else None
                title  = f"H{head}"      if row == 0 else ""
                _draw_cell(axes[row, col], layer_idx, head,
                           title_str=title, ylabel_str=ylabel)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.4, pad=0.005, aspect=40)
    cbar.set_label("Median Attention Gini", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    src_tag   = f" — {prompt_source}" if prompt_source else " — All prompts"
    layer_tag = (f" — Layer {layers[0]}" if n_l == 1
                 else f" — Layers {layers[0]}–{layers[-1]}")
    fig.suptitle(
        f"Attention Gini per Head{src_tag}{layer_tag}",
        fontsize=10, fontweight="bold"
    )

    if save:
        safe = (f"gini_boxplot"
                + (f"_{prompt_source[:4]}" if prompt_source else "")
                + (f"_L{layers[0]}" if n_l == 1 else f"_L{layers[0]}-{layers[-1]}"))
        plt.savefig(f"{safe}.png", dpi=180, bbox_inches="tight")

    display(fig)
    plt.close(fig)


def _build_corr_df_v2(df, features):
    base_idx   = ["layer_idx", "head_idx"]
    prompt_col = "prompt_idx" if "prompt_idx" in df.columns else None

    model_dep_feats = [f for f in features if FEATURE_DEDUP.get(f) is not None and f in df.columns]
    other_feats     = [f for f in features if FEATURE_DEDUP.get(f) is None        and f in df.columns]

    base_hd = (df[base_idx].drop_duplicates()
                            .sort_values(base_idx)
                            .reset_index(drop=True))
    df_A = base_hd.copy()
    for f in model_dep_feats:
        dedup = FEATURE_DEDUP[f]
        tmp   = df.copy()
        if "kv_head" in dedup:
            if "kv_head" not in tmp.columns:
                tmp["kv_head"] = tmp["head_idx"] // 4
            mapping = (tmp.drop_duplicates(subset=["layer_idx", "kv_head"])
                          [["layer_idx", "kv_head", f]])
            df_A["kv_head"] = df_A["head_idx"] // 4
            df_A = df_A.merge(mapping, on=["layer_idx", "kv_head"], how="left")
        else:
            mapping = (tmp.drop_duplicates(subset=base_idx)[base_idx + [f]])
            df_A = df_A.merge(mapping, on=base_idx, how="left")
    if "kv_head" in df_A.columns:
        df_A = df_A.drop(columns=["kv_head"])
    df_A = df_A.set_index(base_idx)

    if other_feats:
        prompt_cols = base_idx + ([prompt_col] if prompt_col else []) + other_feats
        df_B = df[[c for c in prompt_cols if c in df.columns]].copy().reset_index(drop=True)
        for f in model_dep_feats:
            model_col = df_A[[f]].reset_index()
            df_B = df_B.merge(model_col, on=base_idx, how="left")
        idx_cols = base_idx + ([prompt_col] if prompt_col else [])
        df_B = df_B.set_index(idx_cols)
    else:
        df_B = None

    return df_A, df_B


def _build_corr_df_quartile_v2(df, features):
    base_idx   = ["layer_idx", "head_idx"]
    prompt_col = "prompt_idx" if "prompt_idx" in df.columns else None

    model_dep_feats = [f for f in features if FEATURE_DEDUP.get(f) is not None and f in df.columns]
    other_feats     = [f for f in features if FEATURE_DEDUP.get(f) is None        and f in df.columns]

    base_hd = (df[base_idx].drop_duplicates()
                            .sort_values(base_idx)
                            .reset_index(drop=True))
    df_A = base_hd.copy()

    for f in model_dep_feats:
        dedup = FEATURE_DEDUP[f]
        tmp   = df.copy()
        if "kv_head" in dedup:
            if "kv_head" not in tmp.columns:
                tmp["kv_head"] = tmp["head_idx"] // 4
            mapping = (tmp.drop_duplicates(subset=["layer_idx", "kv_head"])
                          [["layer_idx", "kv_head", f]])
            df_A["kv_head"] = df_A["head_idx"] // 4
            df_A = df_A.merge(mapping, on=["layer_idx", "kv_head"], how="left")
        else:
            mapping = (tmp.drop_duplicates(subset=base_idx)[base_idx + [f]])
            df_A = df_A.merge(mapping, on=base_idx, how="left")

    if "kv_head" in df_A.columns:
        df_A = df_A.drop(columns=["kv_head"])
    df_A = df_A.set_index(base_idx)

    if other_feats:
        prompt_cols = base_idx + ([prompt_col] if prompt_col else []) + other_feats
        df_B = df[[c for c in prompt_cols if c in df.columns]].copy().reset_index(drop=True)
        for f in model_dep_feats:
            model_col = df_A[[f]].reset_index()
            df_B = df_B.merge(model_col, on=base_idx, how="left")
        idx_cols = base_idx + ([prompt_col] if prompt_col else [])
        df_B = df_B.set_index(idx_cols)
    else:
        df_B = None

    return df_A, df_B


def _build_corr_matrix(df_A, df_B, features, method):
    model_dep = [f for f in features if FEATURE_DEDUP.get(f) is not None]
    other     = [f for f in features if FEATURE_DEDUP.get(f) is None]

    n = len(features)
    feat_idx = {f: i for i, f in enumerate(features)}
    corr_mat = np.full((n, n), np.nan)

    if model_dep and df_A is not None:
        sub = df_A[[f for f in model_dep if f in df_A.columns]].dropna(how="all")
        if sub.shape[0] >= 2:
            c = sub.corr(method=method)
            for fi in model_dep:
                for fj in model_dep:
                    if fi in c.index and fj in c.columns:
                        corr_mat[feat_idx[fi], feat_idx[fj]] = c.loc[fi, fj]

    if df_B is not None:
        all_in_B = [f for f in features if f in df_B.columns]
        sub = df_B[all_in_B].dropna(how="all")
        if sub.shape[0] >= 2:
            c = sub.corr(method=method)
            for fi in all_in_B:
                for fj in all_in_B:
                    if fi in c.index and fj in c.columns:
                        i, j = feat_idx[fi], feat_idx[fj]
                        if not (fi in model_dep and fj in model_dep):
                            corr_mat[i, j] = c.loc[fi, fj]

    for i in range(n):
        for j in range(i):
            if not np.isnan(corr_mat[i, j]) and np.isnan(corr_mat[j, i]):
                corr_mat[j, i] = corr_mat[i, j]
            elif not np.isnan(corr_mat[j, i]) and np.isnan(corr_mat[i, j]):
                corr_mat[i, j] = corr_mat[j, i]
            elif not np.isnan(corr_mat[i, j]) and not np.isnan(corr_mat[j, i]):
                v = (corr_mat[i, j] + corr_mat[j, i]) / 2
                corr_mat[i, j] = corr_mat[j, i] = v
    np.fill_diagonal(corr_mat, 1.0)

    return pd.DataFrame(corr_mat, index=features, columns=features)


def plot_correlation_matrices(df, taxonomy=TAXONOMY,
                               method="spearman",
                               exclude_subgroups=None,
                               save=SAVE_PLOTS):
    from IPython.display import display

    features    = _get_all_features(taxonomy, df, exclude_subgroups=exclude_subgroups)
    feat_to_cat = _feat_to_category(taxonomy)

    def _corr_for_subset(df_sub):
        df_A, df_B = _build_corr_df_v2(df_sub, features)
        return _build_corr_matrix(df_A, df_B, features, method)

    corr_all  = _corr_for_subset(df)
    corr_wiki = _corr_for_subset(df[df["prompt_source"] == PTYPE_WIKI])
    corr_rand = _corr_for_subset(df[df["prompt_source"] == PTYPE_RAND])

    kw = dict(feat_order=features, feat_to_cat=feat_to_cat)

    fig1 = _make_plotly_heatmap(corr_all,  f"Correlation Matrix — All     [{method}]", **kw)
    fig2 = _make_plotly_heatmap(corr_wiki, f"Correlation Matrix — Wikitext [{method}]", **kw)
    fig3 = _make_plotly_heatmap(corr_rand, f"Correlation Matrix — Random  [{method}]", **kw)

    for fig, name in [(fig1,"all"), (fig2,"wiki"), (fig3,"rand")]:
        if save:
            fig.write_html(str(out_dir / f"corr_{name}_{method}.html"))
        fig.show()

    return corr_all, corr_wiki, corr_rand


def plot_correlation_matrices_by_quartile(df, taxonomy=TAXONOMY,
                                           method="spearman",
                                           exclude_subgroups=None,
                                           save=SAVE_PLOTS):
    from IPython.display import display

    features    = _get_all_features(taxonomy, df, exclude_subgroups=exclude_subgroups)
    feat_to_cat = _feat_to_category(taxonomy)

    layers   = sorted(df["layer_idx"].unique())
    n_layers = len(layers)
    q_size   = n_layers // 4
    layer_to_q = {l: min(i // q_size, 3) for i, l in enumerate(layers)}

    df = df.copy()
    df["layer_quartile"] = df["layer_idx"].map(layer_to_q)

    quartile_labels = {
        0: f"Q1 early  (layers {layers[0]}–{layers[q_size-1]})",
        1: f"Q2        (layers {layers[q_size]}–{layers[2*q_size-1]})",
        2: f"Q3        (layers {layers[2*q_size]}–{layers[3*q_size-1]})",
        3: f"Q4 late   (layers {layers[3*q_size]}–{layers[-1]})",
    }

    prompt_sources = {
        "Wikitext": PTYPE_WIKI,
        "Random":   PTYPE_RAND,
    }

    results = {}
    for src_name, src_val in prompt_sources.items():
        df_src = df[df["prompt_source"] == src_val]

        for q_idx in range(4):
            df_q    = df_src[df_src["layer_quartile"] == q_idx]
            q_label = quartile_labels[q_idx]

            if df_q.empty:
                print(f"[SKIP] {src_name} / {q_label}: df vuoto")
                continue

            df_A, df_B = _build_corr_df_quartile_v2(df_q, features)

            n_obs_A = df_A.shape[0]
            n_obs_B = df_B.shape[0] if df_B is not None else 0
            if n_obs_A < 3 and n_obs_B < 3:
                print(f"[SKIP] {src_name} / {q_label}: troppo poche osservazioni "
                      f"(A={n_obs_A}, B={n_obs_B})")
                continue

            corr = _build_corr_matrix(df_A, df_B, features, method)

            feats_ord = [f for f in features if f in corr.columns]
            corr = corr.loc[feats_ord, feats_ord]

            title = f"{src_name} — {q_label}  [{method}]"
            fig   = _make_plotly_heatmap(corr, title, feats_ord, feat_to_cat)

            if save:
                safe_name = (f"corr_{src_name.lower()}_{q_idx}_{method}"
                             .replace(" ", "_"))
                fig.write_html(str(out_dir / f"{safe_name}.html"))

            display(fig)
            results[(src_name, q_idx)] = corr

    return results


def plot_rope_vs_attn_scatter(df, save=SAVE_PLOTS):
    """Scatterplot: RoPE features (X) × Attention map features (Y)."""
    from IPython.display import display
    from scipy.stats import theilslopes

    x_feats = [
        "rope_pair_var_Wq",
        "rope_pair_var_Wk",
        "rope_pair_max_ratio_Wq",
        "rope_pair_max_ratio_Wk",
        "rope_freq_com_Wq",
        "rope_freq_com_Wk",
    ]
    y_feats = [
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
    ]

    x_feats = [f for f in x_feats if f in df.columns]
    y_feats = [f for f in y_feats if f in df.columns]

    if not x_feats or not y_feats:
        print("[SKIP] Feature non trovate nel df")
        return

    df_rand   = df[df["prompt_source"] == PTYPE_RAND]
    all_feats = list(dict.fromkeys(x_feats + y_feats))
    df_A, df_B = _build_corr_df_v2(df_rand, all_feats)
    agg = df_B.reset_index() if df_B is not None else df_A.reset_index()

    n_rows = len(y_feats)
    n_cols = len(x_feats)

    n_layers = agg["layer_idx"].nunique() if "layer_idx" in agg.columns else 1
    cmap     = plt.get_cmap("plasma", n_layers)
    norm     = plt.Normalize(vmin=0, vmax=n_layers - 1)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 2.6 * n_rows),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)

    for row, y_f in enumerate(y_feats):
        for col, x_f in enumerate(x_feats):
            ax = axes[row, col]

            sub = agg[[x_f, y_f, "layer_idx"]].dropna()
            if sub.empty:
                ax.set_visible(False)
                continue

            ax.scatter(
                sub[x_f], sub[y_f],
                c=sub["layer_idx"], cmap=cmap, norm=norm,
                s=14, alpha=0.7, linewidths=0,
            )

            if len(sub) > 5:
                res   = theilslopes(sub[y_f].values, sub[x_f].values)
                x_fit = np.array([sub[x_f].min(), sub[x_f].max()])
                ax.plot(x_fit, res.slope * x_fit + res.intercept,
                        color="#cc0000", lw=1.0, ls="--", zorder=4)

                rho = sub[[x_f, y_f]].corr(method="spearman").iloc[0, 1]
                ax.annotate(f"ρ={rho:.2f}",
                            xy=(0.04, 0.92), xycoords="axes fraction",
                            fontsize=6, color="#cc0000",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", ec="none", alpha=0.7))

            if row == n_rows - 1:
                ax.set_xlabel(x_f.replace("_", " "), fontsize=6)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(y_f.replace("_", " "), fontsize=6)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=5)
            ax.grid(True, lw=0.3, alpha=0.35)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.01, aspect=40)
    cbar.set_label("Layer index", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("RoPE features vs Attention Map features — Random subset",
                 fontsize=11, fontweight="bold")

    if save:
        _save_fig(fig, "scatter_rope_vs_attn_random")
    display(fig)
    plt.close(fig)


def plot_scatter_quartile(df, quartile: int = 1, method="spearman",
                          prompt_source=None, save=SAVE_PLOTS):
    """Scatterplot pairwise: target features (Y) × model_dependent features (X)."""
    from IPython.display import display
    from scipy.stats import theilslopes

    assert quartile in (0, 1, 2, 3), "quartile deve essere 0, 1, 2 o 3"

    layers   = sorted(df["layer_idx"].unique())
    n_layers = len(layers)
    q_size   = n_layers // 4

    q_start  = quartile * q_size
    q_end    = (quartile + 1) * q_size if quartile < 3 else n_layers
    q_layers = layers[q_start:q_end]

    df_q = df[df["layer_idx"].isin(q_layers)].copy()

    if prompt_source is not None:
        df_q = df_q[df_q["prompt_source"] == prompt_source]

    q_label = f"Q{quartile + 1}"
    print(f"{q_label} layers: {q_layers[0]}–{q_layers[-1]}  ({len(df_q)} righe)")

    x_feats = [f for f in [
        "rope_pair_var_Wq", "rope_pair_var_Wk",
        "rope_pair_max_ratio_Wq", "rope_pair_max_ratio_Wk",
        "rope_freq_com_Wq", "rope_freq_com_Wk",
    ] if f in df.columns]

    y_feats = [f for f in [
        "diagonal_mass_1", "diagonal_mass_5",
        "diagonal_mass_1_shifted_1", "diagonal_mass_1_shifted_2",
        "diagonal_mass_1_shifted_3", "diagonal_mass_1_shifted_4",
        "effective_rank_A", "r95_A",
    ] if f in df.columns]

    all_feats    = list(dict.fromkeys(x_feats + y_feats))
    df_A, df_B   = _build_corr_df_quartile_v2(df_q, all_feats)

    plot_df = df_B.reset_index()
    if "layer_idx" not in plot_df.columns:
        plot_df = plot_df.merge(
            df_q[["layer_idx", "head_idx"]].drop_duplicates(),
            on=["layer_idx", "head_idx"], how="left"
        )

    n_y = len(y_feats)
    n_x = len(x_feats)

    cmap = plt.get_cmap("plasma", len(q_layers))
    norm = plt.Normalize(vmin=q_layers[0], vmax=q_layers[-1])

    fig, axes = plt.subplots(
        n_y, n_x,
        figsize=(2.8 * n_x, 2.4 * n_y),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for row, y_f in enumerate(y_feats):
        for col, x_f in enumerate(x_feats):
            ax = axes[row, col]

            cols_needed = [c for c in [x_f, y_f, "layer_idx"] if c in plot_df.columns]
            sub      = plot_df[cols_needed].copy()
            mask     = sub[[x_f, y_f]].notna().all(axis=1)
            sub_pair = sub[mask]

            if sub_pair.empty:
                ax.set_visible(False)
                continue

            ax.scatter(
                sub_pair[x_f], sub_pair[y_f],
                c=sub_pair["layer_idx"] if "layer_idx" in sub_pair.columns else "#888",
                cmap=cmap, norm=norm,
                s=10, alpha=0.6, linewidths=0,
            )

            if len(sub_pair) > 5:
                res   = theilslopes(sub_pair[y_f].values, sub_pair[x_f].values)
                x_fit = np.array([sub_pair[x_f].min(), sub_pair[x_f].max()])
                ax.plot(x_fit, res.slope * x_fit + res.intercept,
                        color="#cc0000", lw=1.0, ls="--", zorder=4)

                rho       = sub_pair[[x_f, y_f]].corr(method=method).iloc[0, 1]
                color_ann = "#cc0000" if abs(rho) > 0.4 else "#555"
                ax.annotate(f"ρ={rho:.2f}",
                            xy=(0.05, 0.90), xycoords="axes fraction",
                            fontsize=6, color=color_ann, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc="white", ec="none", alpha=0.75))

            if row == n_y - 1:
                ax.set_xlabel(x_f.replace("_", " "), fontsize=5, labelpad=2)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(y_f.replace("_", " "), fontsize=5, labelpad=2)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=5)
            ax.grid(True, lw=0.3, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.4, pad=0.005, aspect=40)
    cbar.set_label(f"Layer index ({q_label})", fontsize=8)
    cbar.set_ticks(q_layers[::2])
    cbar.ax.tick_params(labelsize=7)

    src_tag = f" — {prompt_source}" if prompt_source else " — All prompts"
    fig.suptitle(
        f"{q_label} layers {q_layers[0]}–{q_layers[-1]}: "
        f"Model-dependent (X) × Target (Y){src_tag}",
        fontsize=10, fontweight="bold",
    )

    if save:
        safe_src = ("_" + prompt_source[:4]) if prompt_source else ""
        _save_fig(fig, f"scatter_q{quartile + 1}{safe_src}")
    display(fig)
    plt.close(fig)


def compute_layerwise_rank_diagonal_corr(
    df,
    method: str = "spearman",
    prompt_source: str | None = None,
) -> pd.DataFrame:
    """Compute layer-wise correlations between rank features and diagonal masses."""
    if prompt_source is not None:
        df = df[df["prompt_source"] == prompt_source].copy()

    x_feats = [f for f in [
        "rope_pair_var_Wq", "rope_pair_var_Wk",
        "rope_pair_max_ratio_Wq", "rope_pair_max_ratio_Wk",
        "rope_freq_com_Wq", "rope_freq_com_Wk",
    ] if f in df.columns]

    y_feats = [f for f in [
        "diagonal_mass_1", "diagonal_mass_5",
        "diagonal_mass_1_shifted_1", "diagonal_mass_1_shifted_2",
        "diagonal_mass_1_shifted_3", "diagonal_mass_1_shifted_4",
    ] if f in df.columns]

    base_idx   = ["layer_idx", "head_idx"]
    prompt_col = "prompt_idx" if "prompt_idx" in df.columns else None
    layers     = sorted(df["layer_idx"].unique())

    records = []

    for layer in layers:
        df_l = df[df["layer_idx"] == layer]

        spine_cols = base_idx + ([prompt_col] if prompt_col else []) + y_feats
        spine = df_l[[c for c in spine_cols if c in df_l.columns]].copy().reset_index(drop=True)

        for x_f in x_feats:
            if x_f not in df_l.columns:
                continue
            dedup = FEATURE_DEDUP.get(x_f, None)
            tmp   = df_l.copy()

            if dedup is not None and "kv_head" in dedup:
                if "kv_head" not in tmp.columns:
                    tmp["kv_head"] = tmp["head_idx"] // 4
                mapping = (tmp.drop_duplicates(subset=["layer_idx", "kv_head"])
                              [["head_idx", "kv_head", x_f]])
                spine["kv_head"] = spine["head_idx"] // 4
                spine = spine.merge(mapping[["kv_head", x_f]],
                                    on="kv_head", how="left").drop(columns=["kv_head"])
            else:
                mapping = (tmp.drop_duplicates(subset=base_idx)
                              [["head_idx", x_f]])
                spine = spine.merge(mapping, on="head_idx", how="left")

        for x_f in x_feats:
            if x_f not in spine.columns:
                continue
            for y_f in y_feats:
                if y_f not in spine.columns:
                    continue

                mask = spine[[x_f, y_f]].notna().all(axis=1)
                sub  = spine[mask]

                if len(sub) < 5:
                    rho, pval = np.nan, np.nan
                elif method == "spearman":
                    rho, pval = spearmanr(sub[x_f].values, sub[y_f].values)
                elif method == "pearson":
                    rho  = sub[[x_f, y_f]].corr(method="pearson").iloc[0, 1]
                    pval = np.nan
                else:
                    raise ValueError(f"method '{method}' non supportato")

                records.append({
                    "layer_idx": layer,
                    "x_feat":    x_f,
                    "y_feat":    y_f,
                    "rho":       rho,
                    "pval":      pval,
                    "n_pairs":   len(sub),
                })

    df_records = pd.DataFrame(records)

    table = (df_records
             .pivot_table(index=["x_feat", "y_feat"],
                          columns="layer_idx",
                          values="rho",
                          aggfunc="first")
             .rename_axis("layer", axis=1))

    row_order = [(xf, yf) for xf in x_feats for yf in y_feats
                 if (xf, yf) in table.index]
    table = table.loc[row_order]

    return table, df_records


def plot_scatter_single_layer(
    df,
    layer_idx: int,
    x_feats: list,
    y_feats: list,
    method: str = "spearman",
    prompt_source: str | None = None,
    save=False,
):
    """Generate scatterplots for a single layer."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, theilslopes
    from IPython.display import display

    df_l = df[df["layer_idx"] == layer_idx].copy()
    if prompt_source is not None:
        df_l = df_l[df_l["prompt_source"] == prompt_source]

    if df_l.empty:
        print(f"Nessun dato per layer_idx={layer_idx} e prompt_source={prompt_source}")
        return

    base_idx   = ["layer_idx", "head_idx"]
    prompt_col = "prompt_idx" if "prompt_idx" in df_l.columns else None

    spine_cols = base_idx + ([prompt_col] if prompt_col else []) + [y for y in y_feats if y in df_l.columns]
    spine = df_l[[c for c in spine_cols if c in df_l.columns]].copy().reset_index(drop=True)

    for x_f in x_feats:
        if x_f not in df_l.columns:
            continue
        dedup = FEATURE_DEDUP.get(x_f, None)
        tmp   = df_l.copy()

        if dedup is not None and "kv_head" in dedup:
            if "kv_head" not in tmp.columns:
                tmp["kv_head"] = tmp["head_idx"] // 4
            mapping = (tmp.drop_duplicates(subset=["layer_idx", "kv_head"])
                          [["kv_head", x_f]])
            spine["kv_head"] = spine["head_idx"] // 4
            spine = spine.merge(mapping, on="kv_head", how="left").drop(columns=["kv_head"])
        else:
            mapping = (tmp.drop_duplicates(subset=base_idx)
                          [["head_idx", x_f]])
            spine = spine.merge(mapping, on="head_idx", how="left")

    valid_x = [x for x in x_feats if x in spine.columns]
    valid_y = [y for y in y_feats if y in spine.columns]

    n_x, n_y = len(valid_x), len(valid_y)
    if n_x == 0 or n_y == 0:
        print("Feature non trovate nel DataFrame.")
        return

    fig, axes = plt.subplots(
        n_y, n_x,
        figsize=(3.0 * n_x, 2.6 * n_y),
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    heads = sorted(spine["head_idx"].unique())
    cmap = plt.get_cmap("viridis", len(heads))
    norm = plt.Normalize(vmin=min(heads), vmax=max(heads))

    for row, y_f in enumerate(valid_y):
        for col, x_f in enumerate(valid_x):
            ax = axes[row, col]

            mask = spine[[x_f, y_f]].notna().all(axis=1)
            sub = spine[mask]

            if sub.empty:
                ax.set_visible(False)
                continue

            ax.scatter(
                sub[x_f], sub[y_f],
                c=sub["head_idx"], cmap=cmap, norm=norm,
                s=15, alpha=0.6, linewidths=0, zorder=2
            )

            if len(sub) > 5:
                res = theilslopes(sub[y_f].values, sub[x_f].values)
                x_fit = np.array([sub[x_f].min(), sub[x_f].max()])
                ax.plot(x_fit, res.slope * x_fit + res.intercept,
                        color="#cc0000", lw=1.2, ls="--", zorder=4)

                if method == "spearman":
                    rho, pval = spearmanr(sub[x_f], sub[y_f])
                else:
                    rho = sub[[x_f, y_f]].corr(method="pearson").iloc[0, 1]
                    pval = 1.0

                color_ann = "#cc0000" if abs(rho) > 0.4 and pval < 0.05 else "#555"
                sig_star = "*" if pval < 0.05 else ""
                ax.annotate(f"ρ={rho:.2f}{sig_star}",
                            xy=(0.05, 0.88), xycoords="axes fraction",
                            fontsize=8, color=color_ann, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                            zorder=5)

            if row == n_y - 1:
                ax.set_xlabel(x_f.replace("_", " "), fontsize=8)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(y_f.replace("_", " "), fontsize=8)
            else:
                ax.set_yticklabels([])

            ax.grid(True, lw=0.3, alpha=0.4, zorder=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.01, aspect=30)
    cbar.set_label("Head Index", fontsize=9)

    src_title = f" — {prompt_source}" if prompt_source else ""
    fig.suptitle(f"Scatterplot @ Layer {layer_idx}{src_title}", fontsize=12, fontweight="bold")

    if save:
        plt.savefig(f"scatter_layer_{layer_idx}_{prompt_source or 'all'}.png", dpi=200, bbox_inches="tight")

    display(fig)
    plt.close(fig)


def plot_layerwise_rank_diagonal_corr(
    records_wiki, records_rand,
    x_feats_filter=None,
    y_feats_filter=None,
    save=SAVE_PLOTS,
):
    """Line plots of rho vs layer_idx for each feature pair."""
    from IPython.display import display

    def _filter(rec):
        r = rec.copy()
        if x_feats_filter:
            r = r[r["x_feat"].isin(x_feats_filter)]
        if y_feats_filter:
            r = r[r["y_feat"].isin(y_feats_filter)]
        return r

    rw = _filter(records_wiki)
    rr = _filter(records_rand)

    x_feats = [f for f in (x_feats_filter or records_wiki["x_feat"].unique())
               if f in rw["x_feat"].values or f in rr["x_feat"].values]
    y_feats = [f for f in (y_feats_filter or records_wiki["y_feat"].unique())
               if f in rw["y_feat"].values or f in rr["y_feat"].values]

    n_x = len(x_feats)
    n_y = len(y_feats)

    fig, axes = plt.subplots(
        n_y, n_x,
        figsize=(3.5 * n_x, 2.8 * n_y),
        constrained_layout=True,
        sharey="row",
        sharex=True,
    )
    axes = np.atleast_2d(axes)

    layers_all = sorted(set(rw["layer_idx"].tolist() + rr["layer_idx"].tolist()))

    n_l    = len(layers_all)
    q_size = n_l // 4
    q_boundaries = [
        (layers_all[0],          layers_all[q_size - 1],   "#f0f4ff"),
        (layers_all[q_size],     layers_all[2*q_size - 1], "#fff4e0"),
        (layers_all[2*q_size],   layers_all[3*q_size - 1], "#f0fff4"),
        (layers_all[3*q_size],   layers_all[-1],            "#fff0f4"),
    ]

    for row, y_f in enumerate(y_feats):
        for col, x_f in enumerate(x_feats):
            ax = axes[row, col]

            for l_start, l_end, color in q_boundaries:
                ax.axvspan(l_start - 0.5, l_end + 0.5, color=color, alpha=0.4, zorder=0)

            ax.axhspan(-0.2, 0.2, color="#aaa", alpha=0.10, zorder=1)
            ax.axhline(0, color="#999", lw=0.6, ls="--", zorder=2)

            sub_w = rw[(rw["x_feat"] == x_f) & (rw["y_feat"] == y_f)].sort_values("layer_idx")
            if not sub_w.empty:
                ax.plot(sub_w["layer_idx"], sub_w["rho"],
                        color="#1a6faf", lw=1.8, marker="o", ms=3.5,
                        label="Wikitext", zorder=4)
                sig_w = sub_w[sub_w["pval"] < 0.05]
                if not sig_w.empty:
                    ax.scatter(sig_w["layer_idx"], sig_w["rho"],
                               color="#1a6faf", s=25, zorder=5, edgecolors="white", lw=0.5)

            sub_r = rr[(rr["x_feat"] == x_f) & (rr["y_feat"] == y_f)].sort_values("layer_idx")
            if not sub_r.empty:
                ax.plot(sub_r["layer_idx"], sub_r["rho"],
                        color="#c0392b", lw=1.8, marker="s", ms=3.0,
                        ls="--", label="Random", zorder=4)
                sig_r = sub_r[sub_r["pval"] < 0.05]
                if not sig_r.empty:
                    ax.scatter(sig_r["layer_idx"], sig_r["rho"],
                               color="#c0392b", s=20, zorder=5, edgecolors="white", lw=0.5)

            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.tick_params(labelsize=6)
            ax.grid(True, lw=0.3, alpha=0.3, zorder=1)

            if row == 0:
                ax.set_title(x_f.replace("_", "\n"), fontsize=7, fontweight="bold", pad=4)
            if col == 0:
                ax.set_ylabel(y_f.replace("_", " "), fontsize=6)
            if row == n_y - 1:
                ax.set_xlabel("Layer", fontsize=6)

            if row == 0 and col == 0:
                ax.legend(fontsize=6, framealpha=0.8, loc="lower right")

    for i, (l_start, l_end, _) in enumerate(q_boundaries):
        mid = (l_start + l_end) / 2
        axes[0, 0].text(mid, 1.07, f"Q{i+1}",
                        ha="center", va="bottom", fontsize=6,
                        color="#555", transform=axes[0, 0].get_xaxis_transform())

    fig.suptitle(
        "Layer-wise Spearman ρ:  rank(W) & r95(W)  ×  diagonal mass",
        fontsize=11, fontweight="bold",
    )

    if save:
        _save_fig(fig, "layerwise_rank_diagonal_corr")
    display(fig)
    plt.close(fig)


def plot_target_variability_per_head(
    df,
    target_feat: str = "diagonal_mass_1",
    prompt_source: str | None = None,
    layer: int | list | None = None,
    figsize_per_cell: tuple = (1.4, 1.8),
    save: bool = False,
):
    if prompt_source is not None:
        df = df[df["prompt_source"] == prompt_source].copy()

    if target_feat not in df.columns:
        print(f"Feature '{target_feat}' non trovata nel DataFrame.")
        return

    all_layers = sorted(df["layer_idx"].unique())
    if layer is None:
        layers = all_layers
    elif isinstance(layer, int):
        if layer not in all_layers:
            print(f"Layer {layer} non trovato. Disponibili: {all_layers}")
            return
        layers = [layer]
    elif isinstance(layer, list):
        layers = [l for l in layer if l in all_layers]
        if not layers:
            print(f"Nessuno dei layer specificati trovato. Disponibili: {all_layers}")
            return
    else:
        raise ValueError("'layer' deve essere None, int o list[int]")

    heads = sorted(df["head_idx"].unique())
    n_h   = len(heads)
    n_l   = len(layers)

    single_layer_mode = (n_l == 1)
    if single_layer_mode:
        n_cols = 4
        n_rows = int(np.ceil(n_h / n_cols))
    else:
        n_cols = n_h
        n_rows = n_l

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        constrained_layout=True,
        sharey=True,
        sharex=False,
    )
    axes = np.atleast_2d(axes)

    medians_global = (df.groupby(["layer_idx", "head_idx"])[target_feat]
                        .median().dropna())
    vmin, vmax = medians_global.min(), medians_global.max()
    cmap = plt.get_cmap("RdBu_r")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rng  = np.random.default_rng(42)

    if single_layer_mode:
        layer_idx = layers[0]
        for idx, head in enumerate(heads):
            row, col = divmod(idx, n_cols)
            ax  = axes[row, col]
            sub = df[(df["layer_idx"] == layer_idx) & (df["head_idx"] == head)][target_feat].dropna()

            if sub.empty:
                ax.set_visible(False)
                continue

            median_val = sub.median()
            color      = cmap(norm(median_val))

            ax.boxplot(
                sub.values, widths=0.5, patch_artist=True, notch=False, showfliers=False,
                medianprops=dict(color="#333", lw=1.5),
                whiskerprops=dict(color="#777", lw=0.8),
                capprops=dict(color="#777", lw=0.8),
                boxprops=dict(facecolor=color, alpha=0.75, linewidth=0.5),
            )
            jitter = rng.uniform(-0.18, 0.18, size=len(sub))
            ax.scatter(np.ones(len(sub)) + jitter, sub.values,
                       s=8, alpha=0.6, color=color,
                       edgecolors="#333", linewidths=0.25, zorder=3)

            ax.set_title(f"Head {head}", fontsize=8, pad=3)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=6)
            ax.grid(True, axis="y", lw=0.3, alpha=0.4)

        for idx in range(len(heads), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

    else:
        for row, layer_idx in enumerate(layers):
            for col, head in enumerate(heads):
                ax  = axes[row, col]
                sub = df[(df["layer_idx"] == layer_idx) & (df["head_idx"] == head)][target_feat].dropna()

                if sub.empty:
                    ax.set_visible(False)
                    continue

                median_val = sub.median()
                color      = cmap(norm(median_val))

                ax.boxplot(
                    sub.values, widths=0.5, patch_artist=True, notch=False, showfliers=False,
                    medianprops=dict(color="#333", lw=1.5),
                    whiskerprops=dict(color="#777", lw=0.8),
                    capprops=dict(color="#777", lw=0.8),
                    boxprops=dict(facecolor=color, alpha=0.75, linewidth=0.5),
                )
                jitter = rng.uniform(-0.18, 0.18, size=len(sub))
                ax.scatter(np.ones(len(sub)) + jitter, sub.values,
                           s=6, alpha=0.55, color=color,
                           edgecolors="#333", linewidths=0.2, zorder=3)

                ax.set_xticks([])
                ax.tick_params(axis="y", labelsize=5)
                ax.grid(True, axis="y", lw=0.3, alpha=0.4)

                if col == 0:
                    ax.set_ylabel(f"L{layer_idx}", fontsize=6, rotation=0,
                                  labelpad=18, va="center")
                if row == 0:
                    ax.set_title(f"H{head}", fontsize=6, pad=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.4, pad=0.005, aspect=40)
    cbar.set_label(f"Median {target_feat.replace('_', ' ')}", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    src_tag   = f" — {prompt_source}" if prompt_source else " — All prompts"
    layer_tag = (f" — Layer {layers[0]}" if n_l == 1
                 else f" — Layers {layers[0]}–{layers[-1]}")
    fig.suptitle(
        f"Variabilità inter-prompt: {target_feat.replace('_', ' ')}{src_tag}{layer_tag}",
        fontsize=10, fontweight="bold"
    )

    if save:
        safe = (target_feat
                + (f"_{prompt_source[:4]}" if prompt_source else "")
                + (f"_L{layers[0]}" if n_l == 1 else f"_L{layers[0]}-{layers[-1]}"))
        plt.savefig(f"boxplot_head_variability_{safe}.png", dpi=180, bbox_inches="tight")

    display(fig)
    plt.close(fig)


def plot_interactive_3d_sim_vs_mass(df, save_html=True):
    df_plot = df.dropna(subset=["q_sim_consecutive", "k_sim_consecutive", "diagonal_mass_1_shifted_1"]).copy()

    hover_cols = ["layer_idx", "head_idx"]
    if "prompt_idx" in df_plot.columns:
        hover_cols.append("prompt_idx")
    elif "prompt_source" in df_plot.columns:
        hover_cols.append("prompt_source")

    fig = px.scatter_3d(
        df_plot,
        x="q_sim_consecutive",
        y="k_sim_consecutive",
        z="diagonal_mass_1",
        color="layer_idx",
        hover_data=hover_cols,
        color_continuous_scale="Plasma",
        opacity=0.6,
        title="<b>Attention Heads: Q-Sim vs K-Sim vs Diagonal Mass</b><br><sup>Ogni punto è una combinazione (layer, head, prompt)</sup>"
    )

    fig.update_traces(
        marker=dict(size=2, line=dict(width=0))
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Q-Sim Consecutive (Q)",
            yaxis_title="K-Sim Consecutive (K)",
            zaxis_title="Diagonal Mass 1",
            xaxis=dict(backgroundcolor="#f8f9fa", gridcolor="white", showbackground=True, zerolinecolor="white"),
            yaxis=dict(backgroundcolor="#f8f9fa", gridcolor="white", showbackground=True, zerolinecolor="white"),
            zaxis=dict(backgroundcolor="#f8f9fa", gridcolor="white", showbackground=True, zerolinecolor="white"),
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    if save_html:
        fig.write_html("3d_scatter_q_k_sim_vs_mass.html")
        print("Grafico salvato come '3d_scatter_q_k_sim_vs_mass.html'")

    fig.show()


def scatter_fit(fig, ax, x, y, layer_depth, xlabel, ylabel, title):
    from scipy import stats

    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc, lc = x[mask], y[mask], layer_depth[mask]

    z = np.polyfit(xc, yc, 1)
    xl = np.linspace(xc.min(), xc.max(), 200)
    r, p = stats.pearsonr(xc, yc)

    sc = ax.scatter(xc, yc, c=lc, cmap='viridis', alpha=0.7, s=30, edgecolor='none')
    ax.plot(xl, np.poly1d(z)(xl), 'r-', lw=2, label=f'Linear fit  r={r:.3f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Layer Depth', rotation=270, labelpad=15)
