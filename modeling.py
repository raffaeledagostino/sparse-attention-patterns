

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Plot style ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     "#333",
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.color":         "#CCCCCC",
    "grid.linewidth":     0.5,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "axes.titlesize":     9,
    "axes.titleweight":   "bold",
    "axes.titlepad":      6,
    "font.family":        "DejaVu Sans",
})

C_MODDEP   = "#2E86AB"
C_INPUTDEP = "#E84855"

# ── Save helpers ──────────────────────────────────────────────────────────────

def _safe_name(s: str) -> str:
    return (s.lower()
             .replace(" ", "_").replace("(", "").replace(")", "")
             .replace(",", "").replace("/", "_").replace("—", "")
             .replace("&", "").replace("__", "_").strip("_"))


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

# ── Model configuration ───────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    key:            str
    label:          str
    ptype_wiki:     str
    ptype_fineweb:  str
    gqa_ratio:      int
    n_heads:        int
    n_kv_heads:     int
    default_target: str = "sink_mass_token_0"


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "mistral_7b": ModelConfig(
        key           = "mistral_7b",
        label         = "Mistral-7B-Instruct-v0.3",
        ptype_wiki    = "wikitext_wikitext-103-raw-v1_train",
        ptype_fineweb = "fineweb-edu_sample-10BT_train_stream",
        gqa_ratio     = 4,
        n_heads       = 32,
        n_kv_heads    = 8,
    ),
    "qwen3_4b": ModelConfig(
        key           = "qwen3_4b",
        label         = "Qwen3-4B",
        ptype_wiki    = "wikitext_wikitext-103-raw-v1_train",
        ptype_fineweb = "fineweb-edu_sample-10BT_train_stream",
        gqa_ratio     = 4,
        n_heads       = 32,
        n_kv_heads    = 8,
    ),
}

# ── Feature / target taxonomy ─────────────────────────────────────────────────

MODEL_DEP_FEATURES: List[str] = [
    "effective_rank_Wq", "r95_Wq",
    "effective_rank_Wk", "r95_Wk",
    "effective_rank_Wv", "r95_Wv",
    "gini_left_Wq", "gini_right_Wq",
    "gini_left_Wk", "gini_right_Wk",
    "rope_pair_var_Wq", "rope_pair_var_Wk",
    "rope_pair_max_ratio_Wq", "rope_pair_max_ratio_Wk",
    "rope_freq_com_Wq", "rope_freq_com_Wk",
    "rmsnorm_gamma_norm",
    "compute_WqRWk_alignment_delta_0",
]

# Online features: only q_sim and k_sim for now.
# To add more, extend this list.
INPUT_DEP_FEATURES: List[str] = [
    "q_sim_consecutive",
    "k_sim_consecutive",
]

FEATURE_SETS: Dict[str, List[str]] = {
    "offline": MODEL_DEP_FEATURES,
    "oracle":  MODEL_DEP_FEATURES + INPUT_DEP_FEATURES,
}

ALL_TARGETS: List[str] = [
    "attention_gini",
    "diagonal_mass_1", "diagonal_mass_5",
    "diagonal_mass_1_shifted_1", "diagonal_mass_1_shifted_2",
    "diagonal_mass_1_shifted_3", "diagonal_mass_1_shifted_4",
    "sink_mass_token_0", "sink_mass_token_1", "sink_mass_token_2",
    "sink_mass_token_3", "sink_mass_token_4", "sink_mass_max",
    "look_back",
    "effective_rank_A", "r95_A",
]

# Accumulative run cache — never reset between calls.
# key: (exp_tag, model_key, target, variant)
_RUN_CACHE: Dict[Tuple[str, str, str, str], dict] = {}

# ── Split helpers ─────────────────────────────────────────────────────────────

def split_by_prompt(
    df:        pd.DataFrame,
    val_frac:  float = 0.10,
    test_frac: float = 0.10,
    seed:      int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prompts = np.array(sorted(df["prompt_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(prompts)
    n      = len(prompts)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    test_p  = set(prompts[:n_test])
    val_p   = set(prompts[n_test:n_test + n_val])
    train_p = set(prompts[n_test + n_val:])
    return (
        df[df["prompt_id"].isin(train_p)].copy(),
        df[df["prompt_id"].isin(val_p)].copy(),
        df[df["prompt_id"].isin(test_p)].copy(),
    )


def split_by_head(
    df_agg:    pd.DataFrame,
    val_frac:  float = 0.15,
    test_frac: float = 0.20,
    seed:      int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(df_agg))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n      = len(idx)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))
    return (
        df_agg.iloc[idx[n_test + n_val:]].reset_index(drop=True),
        df_agg.iloc[idx[n_test:n_test + n_val]].reset_index(drop=True),
        df_agg.iloc[idx[:n_test]].reset_index(drop=True),
    )


def split_by_length_stratified(
    df:             pd.DataFrame,
    train_lengths:  Tuple[int, ...] = (64, 128, 256),   
    test_lengths:   Tuple[int, ...] = (512,),
    val_frac:       float = 0.15,
    seed:           int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by prompt length.
    Train/val: prompt_len in train_lengths.
    Test: prompt_len in test_lengths (unseen distribution).
    Validation is sampled stratified by (prompt_len, prompt_source).
    """
    df_test     = df[df["prompt_len"].isin(test_lengths)].copy()
    df_trainval = df[df["prompt_len"].isin(train_lengths)].copy()
    rng = np.random.default_rng(seed)
    val_prompts: set = set()
    for _, grp in df_trainval.groupby(["prompt_len", "prompt_source"]):
        uniq = np.array(sorted(grp["prompt_id"].unique()))
        rng.shuffle(uniq)
        val_prompts.update(uniq[:max(1, int(len(uniq) * val_frac))])
    df_val   = df_trainval[df_trainval["prompt_id"].isin(val_prompts)].copy()
    df_train = df_trainval[~df_trainval["prompt_id"].isin(val_prompts)].copy()
    return df_train, df_val, df_test

# ── Baselines ─────────────────────────────────────────────────────────────────

def head_mean_baseline(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    target:   str,
) -> np.ndarray:
    """
    Exp. 1 — Cross-Prompt baseline.
    For each test head: mean target for that head across other
    prompts in the training set. Fallback: global train mean.
    """
    head_means = df_train.groupby(["layer_idx", "head_idx"])[target].mean()
    
    test_multi_idx = pd.MultiIndex.from_frame(df_test[["layer_idx", "head_idx"]])
    
    pred_baseline = test_multi_idx.map(head_means)
    
    global_mean = df_train[target].mean()
    pred_baseline = pred_baseline.fillna(global_mean)
    
    return pred_baseline.to_numpy()


def nearest_neighbor_baseline(
    df_train:   pd.DataFrame,
    df_test:    pd.DataFrame,
    feat_cols:  List[str],
    target_col: str = "target_median",
) -> np.ndarray:
    """
    Exp. 2 — Cross-Head baseline.
    Exact value from the nearest head in the train set (L2 distance on
    MODEL_DEP_FEATURES normalised by std). No averaging — single nearest neighbour.
    """
    md_cols = [c for c in feat_cols if c in MODEL_DEP_FEATURES and c in df_train.columns]
    if not md_cols:
        return np.full(len(df_test), df_train[target_col].mean())
    X_tr = df_train[md_cols].fillna(0).values
    X_te = df_test[md_cols].fillna(0).values
    mu    = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0) + 1e-8
    X_tr_n = (X_tr - mu) / sigma
    X_te_n = (X_te - mu) / sigma
    # Point estimate (not an average): exact target of the nearest head
    nn_idx = np.array([
        np.argmin(np.linalg.norm(X_tr_n - x, axis=1))
        for x in X_te_n
    ])
    return df_train[target_col].iloc[nn_idx].values


def length_corrected_head_mean_baseline(
    df_train:  pd.DataFrame,
    df_test:   pd.DataFrame,
    target:    str,
) -> np.ndarray:
    """
    Exp. 3 — Length Generalisation baseline.
    Corrects the per-head train mean with an additive length correction:

        pred(h, L_test) = mu_h_train
                        + beta_h * (log(L_test) - mean(log(L_train)))

    beta_h is estimated by OLS on (log(prompt_len), target) per head
    in the train set. Captures systematic target trends with length
    without requiring any observation at L_test.
    """
    log_len_col = "__log_len__"
    df_train = df_train.copy()
    df_train[log_len_col] = np.log(df_train["prompt_len"].clip(lower=2).astype(float))

    # Estimate beta_h per head via OLS (intercept = mu_h)
    def _ols(grp: pd.DataFrame):
        x = grp[log_len_col].values
        y = grp[target].values
        xc = x - x.mean()
        denom = (xc ** 2).sum()
        beta  = (xc * y).sum() / denom if denom > 1e-12 else 0.0
        mu    = y.mean()
        log_len_mean = x.mean()
        return pd.Series({"mu": mu, "beta": beta, "log_len_mean": log_len_mean})

    head_params = (
        df_train.groupby(["layer_idx", "head_idx"])
        .apply(_ols)
        .reset_index()
    )

    df_te = df_test.copy()
    df_te[log_len_col] = np.log(df_te["prompt_len"].clip(lower=2).astype(float))
    merged = df_te.merge(head_params, on=["layer_idx", "head_idx"], how="left")

    global_mu   = df_train[target].mean()
    global_beta = 0.0
    global_log_len_mean = df_train[log_len_col].mean()

    merged["mu"]             = merged["mu"].fillna(global_mu)
    merged["beta"]           = merged["beta"].fillna(global_beta)
    merged["log_len_mean"]   = merged["log_len_mean"].fillna(global_log_len_mean)

    pred = (
        merged["mu"]
        + merged["beta"] * (merged[log_len_col] - merged["log_len_mean"])
    ).values
    return pred

# ── Hyperparameter tuning (Optuna) ────────────────────────────────────────────
#
#
# Rationale for ranges:
#   learning_rate  [0.01, 0.15]: avoids too-low lr (slow, no early-stopping gain)
#                                or too-high lr (underfitting).
#   num_leaves     [15, 255]:    15 is sufficient for small datasets; 255 allows
#                                complexity for harder targets.
#   feature_fraction [0.5, 1.0]: with ~18-20 features, 0.5 gives diversity;
#                                1.0 uses all features.
#   bagging_fraction [0.5, 1.0]: same rationale for row subsampling.
#   min_child_samples [5, 50]:   low bound for cross-head (n~256), upper
#                                bound 50 for cross-prompt (n>>1000).
#   lambda_l1      [1e-4, 10]:   L1 regularisation; log range covers near-zero
#                                to strong penalties.
#   lambda_l2      [1e-4, 10]:   same rationale for L2.
#
# num_boost_round=2000 + early_stopping(60) ensures convergence without
# fixing a suboptimal num_rounds value.

def _lgbm_objective(
    trial:    optuna.Trial,
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    features: List[str],
    target:   str,
    seed:     int,
) -> float:
    params = {
        "objective":          "regression_l1",
        "metric":             "mae",
        "verbose":            -1,
        "seed":               seed,
        "learning_rate":      trial.suggest_float("learning_rate",     0.01,  0.15,  log=True),
        "num_leaves":         trial.suggest_int(  "num_leaves",        15,    255),
        "feature_fraction":   trial.suggest_float("feature_fraction",  0.5,   1.0),
        "bagging_fraction":   trial.suggest_float("bagging_fraction",  0.5,   1.0),
        "bagging_freq":       5,
        "min_child_samples":  trial.suggest_int(  "min_child_samples", 5,     50),
        "lambda_l1":          trial.suggest_float("lambda_l1",         1e-4,  10.0,  log=True),
        "lambda_l2":          trial.suggest_float("lambda_l2",         1e-4,  10.0,  log=True),
    }
    X_tr, y_tr = df_train[features].values, df_train[target].values
    X_va, y_va = df_val[features].values,   df_val[target].values
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    model  = lgb.train(
        params, dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
    )
    return mean_absolute_error(y_va, model.predict(X_va))


def tune_lgbm(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    features: List[str],
    target:   str,
    n_trials: int = 10,
    seed:     int = 42,
) -> dict:
    """Return best hyperparameters found by Optuna TPE."""
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        lambda trial: _lgbm_objective(trial, df_train, df_val, features, target, seed),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study.best_params

# ── Training ──────────────────────────────────────────────────────────────────

def train_lgbm(
    df_train:        pd.DataFrame,
    df_val:          pd.DataFrame,
    features:        List[str],
    target:          str,
    params_override: Optional[dict] = None,
    seed:            int = 42,
) -> lgb.Booster:
    """Train LightGBM with early-stopping on validation. Use params_override if provided."""
    default_params = {
        "objective":          "regression_l1",
        "metric":             "mae",
        "learning_rate":      0.05,
        "num_leaves":         63,
        "feature_fraction":   0.8,
        "bagging_fraction":   0.8,
        "bagging_freq":       5,
        "min_child_samples":  20,
        "lambda_l1":          0.1,
        "lambda_l2":          0.1,
        "verbose":            -1,
        "seed":               seed,
    }
    params = {**default_params, **(params_override or {})}
    X_tr, y_tr = df_train[features].values, df_train[target].values
    X_va, y_va = df_val[features].values,   df_val[target].values
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    return lgb.train(
        params, dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
    )


def train_lgbm_head(
    df_train:        pd.DataFrame,
    df_val:          pd.DataFrame,
    feat_cols:       List[str],
    target_col:      str = "target_median",
    params_override: Optional[dict] = None,
    seed:            int = 42,
) -> Tuple[lgb.Booster, List[str]]:
    """Cross-head version: dataset aggregated per head, small n."""
    feats  = [c for c in feat_cols if c in df_train.columns]
    default_params = {
        "objective":          "regression_l1",
        "metric":             "mae",
        "learning_rate":      0.03,
        "num_leaves":         15,
        "feature_fraction":   0.8,
        "bagging_fraction":   0.8,
        "bagging_freq":       5,
        "min_child_samples":  3,
        "lambda_l1":          0.1,
        "lambda_l2":          0.1,
        "verbose":            -1,
        "seed":               seed,
    }
    params = {**default_params, **(params_override or {})}
    X_tr, y_tr = df_train[feats].fillna(0).values, df_train[target_col].values
    X_va, y_va = df_val[feats].fillna(0).values,   df_val[target_col].values
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feats)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=feats)
    model  = lgb.train(
        params, dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
    )
    return model, feats

# ── Evaluation ────────────────────────────────────────────────────────────────

def _layer_quartile_mae(
    y_te:      np.ndarray,
    pred:      np.ndarray,
    layer_idx: pd.Series,
) -> dict:
    layers = sorted(layer_idx.unique())
    q_size = max(1, len(layers) // 4)
    l_to_q = {l: min(i // q_size, 3) for i, l in enumerate(layers)}
    q_col  = layer_idx.map(l_to_q).values
    return {
        f"MAE_Q{q+1}": round(mean_absolute_error(y_te[q_col == q], pred[q_col == q]), 5)
        for q in range(4) if (q_col == q).sum() > 0
    }


def evaluate(
    model:    lgb.Booster,
    df_test:  pd.DataFrame,
    features: List[str],
    target:   str,
    cfg:      "ModelConfig",  # Assicurati di importare/gestire questa type hint
    df_train: Optional[pd.DataFrame] = None,
    baseline_fn: str = "head_mean",
) -> Tuple[dict, pd.DataFrame]:
    """
    Evaluate the model on the test set.
    """
    X_te = df_test[features].values
    y_te = df_test[target].values
    
    # Assicurati che le prediction siano un array 1D
    pred = model.predict(X_te)
    if pred.ndim > 1:
        pred = pred.squeeze()

    results: dict = {
        "R2":  r2_score(y_te, pred),
        "MAE": mean_absolute_error(y_te, pred),
    }

    if df_train is not None:
        if baseline_fn == "length_corrected":
            pred_base = length_corrected_head_mean_baseline(df_train, df_test, target)
        else:
            pred_base = head_mean_baseline(df_train, df_test, target)
            
        results["MAE_baseline"]    = mean_absolute_error(y_te, pred_base)
        results["MAE_improvement"] = results["MAE_baseline"] - results["MAE"]
        # Evitare divisioni per zero se baseline è perfetta (improbabile, ma best practice)
        results["lift"]            = results["MAE_improvement"] / (results["MAE_baseline"] + 1e-9)

    # Computa MAE diviso per sorgente del prompt
    if "prompt_source" in df_test.columns:
        for src in df_test["prompt_source"].unique():
            mask  = (df_test["prompt_source"] == src).values
            short = src.split("_")[0]
            if mask.sum() > 0:
                results[f"MAE_{short}"] = mean_absolute_error(y_te[mask], pred[mask])

    # Assumo che _layer_quartile_mae restituisca un dict
    # results.update(_layer_quartile_mae(y_te, pred, df_test["layer_idx"]))

    df_out = df_test.copy()
    df_out["pred"]     = pred
    df_out["residual"] = y_te - pred
    
    return results, df_out



def evaluate_head(
    model:      lgb.Booster,
    feats:      List[str],
    df_train:   pd.DataFrame,
    df_test:    pd.DataFrame,
    target_col: str = "target_median",
    label:      str = "",
) -> Tuple[dict, pd.DataFrame]:
    """
    Evaluate the model in the cross-head setting.
    Baseline: nearest_neighbor_baseline (exact L2 nearest head value).
    """
    X_te = df_test[feats].fillna(0).values
    y_te = df_test[target_col].values
    pred = model.predict(X_te)

    pred_nn = nearest_neighbor_baseline(df_train, df_test, feats, target_col)

    mae_model = mean_absolute_error(y_te, pred)
    mae_nn    = mean_absolute_error(y_te, pred_nn)

    out = {
        "label":        label,
        "n_heads_test": len(y_te),
        "R2":           r2_score(y_te, pred),
        "MAE":          mae_model,
        "MAE_nn":       mae_nn,
        "lift_nn":      (mae_nn - mae_model) / max(mae_nn, 1e-12),
    }
    df_out = df_test.copy()
    df_out["pred"]  = pred
    df_out["resid"] = y_te - pred
    return out, df_out

# ── Aggregation (cross-head) ──────────────────────────────────────────────────

def aggregate_per_head(
    df:       pd.DataFrame,
    features: List[str],
    target:   str,
    cfg:      ModelConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate the dataset per (layer_idx, head_idx).
    MODEL_DEP: .first() — model-level constants by definition.
    INPUT_DEP: mean and std across prompts (summarises input-conditional distribution).
    Target: median, mean, std for a robust description.
    """
    model_dep = [f for f in features if f in MODEL_DEP_FEATURES and f in df.columns]
    input_dep = [f for f in features if f in INPUT_DEP_FEATURES and f in df.columns]

    agg_dict: dict = {}
    for f in model_dep:
        agg_dict[f] = (f, "first")
    for f in input_dep:
        agg_dict[f"{f}_mean"] = (f, "mean")
        agg_dict[f"{f}_std"]  = (f, "std")
    agg_dict["target_median"] = (target, "median")
    agg_dict["target_mean"]   = (target, "mean")
    agg_dict["target_std"]    = (target, "std")

    df_agg = (
        df.groupby(["layer_idx", "head_idx"])
        .agg(**agg_dict)
        .reset_index()
    )
    feat_cols = (
        model_dep
        + [f"{f}_mean" for f in input_dep]
        + [f"{f}_std"  for f in input_dep]
    )
    feat_cols = [c for c in feat_cols if c in df_agg.columns]
    return df_agg, feat_cols


def _run_one(
    df_train:    pd.DataFrame,
    df_val:      pd.DataFrame,
    df_test:     pd.DataFrame,
    features:    List[str],
    target:      str,
    var_name:    str,
    cfg:         ModelConfig,
    best_params: Optional[dict] = None,
    baseline_fn: str = "head_mean",
) -> Optional[Tuple[dict, lgb.Booster, List[str], pd.DataFrame]]:
    if target not in df_train.columns:
        return None
    df_tr = df_train.dropna(subset=[target])
    df_va = df_val.dropna(subset=[target])
    df_te = df_test.dropna(subset=[target])
    if len(df_tr) < 50 or len(df_te) < 10:
        return None

    model  = train_lgbm(df_tr, df_va, features, target, params_override=best_params)
    y_te   = df_te[target].values
    pred   = model.predict(df_te[features].values)

    if baseline_fn == "length_corrected":
        pred_b = length_corrected_head_mean_baseline(df_tr, df_te, target)
    else:
        pred_b = head_mean_baseline(df_tr, df_te, target)

    mae      = mean_absolute_error(y_te, pred)
    mae_base = mean_absolute_error(y_te, pred_b)
    r2       = r2_score(y_te, pred)
    lift     = (mae_base - mae) / mae_base

    df_pred = df_te.copy()
    df_pred["pred"]  = pred
    df_pred["resid"] = y_te - pred

    row: dict = {
        "target":       target,
        "variant":      var_name,
        "R2":           round(r2, 4),
        "MAE":          round(mae, 5),
        "MAE_baseline": round(mae_base, 5),
        "lift":         round(lift, 4),
        "best_iter":    model.best_iteration,
    }
    row.update(_layer_quartile_mae(y_te, pred, df_te["layer_idx"]))

    if "prompt_source" in df_te.columns:
        for src in sorted(df_te["prompt_source"].unique()):
            mask  = (df_te["prompt_source"] == src).values
            short = src.split("_")[0]
            if mask.sum() > 0:
                row[f"MAE_{short}"] = round(
                    mean_absolute_error(y_te[mask], pred[mask]), 5)

    return row, model, features, df_pred


def _run_all_targets(
    df_train:    pd.DataFrame,
    df_val:      pd.DataFrame,
    df_test:     pd.DataFrame,
    df:          pd.DataFrame,         
    cfg:         ModelConfig,
    targets:     List[str],
    tune:        bool,
    n_trials:    int,
    seed:        int,
    exp_tag:     str,
    baseline_fn: str = "head_mean",
    out_dir:     Optional[Path] = None,
) -> pd.DataFrame:
    rows: list = []
    valid_targets = [t for t in targets if t in df.columns]
    total = len(valid_targets) * len(FEATURE_SETS)
    done  = 0

    for target in valid_targets:
        # Separate tuning per variant
        best_params_per_variant: Dict[str, Optional[dict]] = {v: None for v in FEATURE_SETS}
        if tune:
            for var_name, features in FEATURE_SETS.items():
                feats_avail = [f for f in features if f in df.columns]
                df_tr_clean = df_train.dropna(subset=[target])
                df_va_clean = df_val.dropna(subset=[target])
                if len(df_tr_clean) < 50:
                    continue
                print(f"     Optuna [{var_name}] {target} ({n_trials} trials)...",
                      end=" ", flush=True)
                best_params_per_variant[var_name] = tune_lgbm(
                    df_tr_clean, df_va_clean, feats_avail, target, n_trials, seed)
                print("done.")

            # Save best_params to disk after tuning all variants for this target
            if out_dir is not None:
                import json
                params_path = out_dir / f"best_params_{_safe_name(target)}.json"
                params_path.parent.mkdir(parents=True, exist_ok=True)
                with open(params_path, "w") as fp:
                    json.dump(best_params_per_variant, fp, indent=2)

        for var_name, features in FEATURE_SETS.items():
            done += 1
            feats_avail = [f for f in features if f in df.columns]
            print(f"[{done:>3}/{total}] {target:<38} {var_name}", end="  ")

            result = _run_one(
                df_train, df_val, df_test,
                feats_avail, target, var_name, cfg,
                best_params=best_params_per_variant.get(var_name),
                baseline_fn=baseline_fn,
            )
            if result is None:
                print("SKIP"); continue

            row, model, feats, df_pred = result
            rows.append(row)
            _RUN_CACHE[(exp_tag, cfg.key, target, var_name)] = {
                "model":      model,
                "feats":      feats,
                "df_pred":    df_pred,
                "df_train":   df_train,
                "best_params": best_params_per_variant.get(var_name),
            }
            print(f"R²={row['R2']:.4f}  MAE={row['MAE']:.5f}  lift={row['lift']:+.1%}")

    if not rows:
        return pd.DataFrame(
            columns=["target", "variant"]
        ).set_index(["target", "variant"])
    return (
        pd.DataFrame(rows)
        .set_index(["target", "variant"])
        .sort_index()
    )

# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_top_features(
    results_cp: dict,
    cfg:        ModelConfig,
    target:     str,
    out_dir:    Optional[Path] = None,
    top_n:      int = 15,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, (var_name, res) in zip(axes, results_cp.items()):
        model    = res["model"]
        features = res["features"]
        df_pred  = res["predictions"]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(df_pred[features].values)
        mean_shap = np.abs(shap_vals).mean(axis=0)
        order     = np.argsort(mean_shap)[::-1][:top_n]
        feat_ord  = [features[i] for i in order]
        shap_ord  = mean_shap[order]
        colors    = [C_MODDEP if f in MODEL_DEP_FEATURES else C_INPUTDEP for f in feat_ord]
        ax.barh(range(len(feat_ord))[::-1], shap_ord,
                color=colors, alpha=0.88, edgecolor="none", height=0.65)
        ax.set_yticks(range(len(feat_ord))[::-1])
        ax.set_yticklabels(feat_ord, fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=9)
        m = res["metrics"]
        ax.set_title(
            f"Variant: {var_name}\n"
            f"R²={m['R2']:.4f}  MAE={m['MAE']:.5f}",
            fontsize=10, fontweight="bold",
        )
        ax.grid(axis="x", lw=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=[
            mpatches.Patch(facecolor=C_MODDEP,   label="model-dependent"),
            mpatches.Patch(facecolor=C_INPUTDEP, label="input-dependent"),
        ], fontsize=8, loc="upper left")
    fig.suptitle(
        f"Top feature importance — Cross-Prompt\n"
        f"Target: {target} | Model: {cfg.label}",
        fontsize=11, fontweight="bold",
    )
    if out_dir is not None:
        _savefig(fig, out_dir / f"shap_{_safe_name(target)}.png")
    plt.show()
    return fig


def plot_shap_run(
    target:  str,
    variant: str = "oracle",
    exp_tag: str = "cross_prompt",
    cfg:     Optional[ModelConfig] = None,
    out_dir: Optional[Path] = None,
    top_n:   int = 15,
) -> Optional[plt.Figure]:
    model_key = cfg.key if cfg else list(MODEL_CONFIGS.keys())[0]
    key = (exp_tag, model_key, target, variant)
    if key not in _RUN_CACHE:
        print(f"Run {key} not in cache. Available: {list(_RUN_CACHE.keys())}")
        return None
    cache     = _RUN_CACHE[key]
    model     = cache["model"]
    feats     = cache["feats"]
    df_te     = cache["df_pred"]
    X         = df_te[feats].fillna(0).values
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    mean_shap = np.abs(shap_vals).mean(axis=0)
    order     = np.argsort(mean_shap)[::-1][:top_n]
    feat_ord  = [feats[i] for i in order]
    shap_ord  = mean_shap[order]
    colors    = [C_MODDEP if f in MODEL_DEP_FEATURES else C_INPUTDEP for f in feat_ord]
    fig, ax   = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.barh(range(len(feat_ord))[::-1], shap_ord,
            color=colors, alpha=0.88, edgecolor="none", height=0.65)
    ax.set_yticks(range(len(feat_ord))[::-1])
    ax.set_yticklabels(feat_ord, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    label = cfg.label if cfg else model_key
    ax.set_title(
        f"SHAP — {exp_tag} | target: {target} | variant: {variant} | {label}\n"
        f"N={len(df_te):,} obs",
        fontsize=10, fontweight="bold",
    )
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_MODDEP,   label="model-dependent"),
        mpatches.Patch(facecolor=C_INPUTDEP, label="input-dependent"),
    ], fontsize=8)
    if out_dir is not None:
        _savefig(fig, out_dir / f"shap_{_safe_name(target)}_{variant}.png")
    plt.show()
    return fig


def plot_shap_crosshead(
    results_ch: dict,
    cfg:        ModelConfig,
    target:     str,
    out_dir:    Optional[Path] = None,
    top_n:      int = 15,
) -> plt.Figure:
    n_variants = len(results_ch)
    fig, axes  = plt.subplots(1, n_variants,
                               figsize=(8 * n_variants, 6), constrained_layout=True)
    if n_variants == 1:
        axes = [axes]
    for ax, (var_name, res) in zip(axes, results_ch.items()):
        model  = res["model"]
        feats  = res["feats"]
        X_test = res["preds"][feats].fillna(0).values
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        mean_shap = np.abs(shap_vals).mean(axis=0)
        order     = np.argsort(mean_shap)[::-1][:top_n]
        feat_ord  = [feats[i] for i in order]
        shap_ord  = mean_shap[order]
        colors    = [C_MODDEP if f in MODEL_DEP_FEATURES else C_INPUTDEP for f in feat_ord]
        ax.barh(range(len(feat_ord))[::-1], shap_ord,
                color=colors, alpha=0.88, edgecolor="none", height=0.65)
        ax.set_yticks(range(len(feat_ord))[::-1])
        ax.set_yticklabels(feat_ord, fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=9)
        m = res["metrics"]
        ax.set_title(
            f"Variant: {var_name}\n"
            f"R²={m['R2']:.4f}  MAE={m['MAE']:.5f}\n"
            f"lift vs NN={m['lift_nn']:+.1%}",
            fontsize=9, fontweight="bold",
        )
        ax.grid(axis="x", lw=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=[
            mpatches.Patch(facecolor=C_MODDEP,   label="model-dependent"),
            mpatches.Patch(facecolor=C_INPUTDEP, label="input-dependent (aggregated)"),
        ], fontsize=8, loc="upper left")
    fig.suptitle(
        f"Experiment 2 — Cross-Head | {cfg.label}\n"
        f"SHAP Feature Importance | Target: {target} (median per head)",
        fontsize=11, fontweight="bold",
    )
    if out_dir is not None:
        _savefig(fig, out_dir / f"shap_crosshead_{_safe_name(target)}.png")
    plt.show()
    return fig


def plot_r2_vs_lift_panels(
    pivot:        pd.DataFrame,
    cfg:          ModelConfig,
    out_dir:      Optional[Path] = None,
    title_suffix: str = "Cross-Prompt",
    lift_col:     str = "lift",         # "lift" | "lift_nn" a seconda dell'exp
) -> plt.Figure:
    GROUPS = {
        "Diagonal targets": {
            "diagonal_mass_1":         "dm1",
            "diagonal_mass_1_shifted_1": "dm1_sh1",
            "diagonal_mass_1_shifted_2": "dm1_sh2",
            "diagonal_mass_1_shifted_3": "dm1_sh3",
            "diagonal_mass_1_shifted_4": "dm1_sh4",
            "diagonal_mass_5":           "dm5",
        },
        "Sink targets": {
            "sink_mass_token_0": "sink_t0",
            "sink_mass_token_1": "sink_t1",
            "sink_mass_token_2": "sink_t2",
            "sink_mass_token_3": "sink_t3",
            "sink_mass_token_4": "sink_t4",
            "sink_mass_max":     "sink_max",
        },
        "Other targets": {
            "attention_gini":    "gini",
            "look_back":         "look_back",
            "effective_rank_A":  "eff_rank_A",
            "r95_A":             "r95_A",
        },
    }
    GROUP_COLORS = {
        "Diagonal targets": ["#1b4f72","#2980b9","#5dade2","#85c1e9","#aed6f1","#d6eaf8"],
        "Sink targets":     ["#784212","#d35400","#e67e22","#f39c12","#f8c471","#fdebd0"],
        "Other targets":    ["#E84855","#6A4C93","#3BB273","#1a7a4a"],
    }
    off_col = f"{lift_col}_offline"
    ora_col = f"{lift_col}_oracle"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    for ax, (group_name, targets_map) in zip(axes, GROUPS.items()):
        colors = GROUP_COLORS[group_name]
        ax.axhline(0,    color="#ccc", lw=0.8, ls="--", zorder=0)
        ax.axvline(0,    color="#ccc", lw=0.8, ls="--", zorder=0)
        ax.axhspan(0.4, 1.05, color="#e8f5ea", alpha=0.35, zorder=0)
        ax.text(0.02, 0.42, "lift > 40%", fontsize=7, color="#3a7a45",
                alpha=0.7, transform=ax.get_xaxis_transform())
        handles_legend = []
        for i, (target, label) in enumerate(targets_map.items()):
            if target not in pivot.index:
                continue
            color = colors[i % len(colors)]
            r2_off   = pivot.loc[target, "R2_offline"]
            r2_ora   = pivot.loc[target, "R2_oracle"]
            lift_off = pivot.loc[target, off_col]
            lift_ora = pivot.loc[target, ora_col]
            ax.annotate("",
                xy=(r2_ora, lift_ora), xytext=(r2_off, lift_off),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                mutation_scale=8),
                zorder=2)
            ax.scatter(r2_off, lift_off, s=70, facecolors="none",
                       edgecolors=color, linewidths=1.8, zorder=3)
            ax.scatter(r2_ora, lift_ora, s=70, color=color,
                       edgecolors="white", linewidths=0.5, zorder=4)
            ax.annotate(label, xy=(r2_ora, lift_ora),
                        xytext=(5, 3), textcoords="offset points",
                        fontsize=8, color="#222")
            handles_legend.append(mpatches.Patch(color=color, label=label))
        ax.set_title(group_name, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("$R^2$", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Lift vs baseline", fontsize=9)
        ax.set_xlim(-0.20, 1.08)
        ax.set_ylim(-0.15, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
        ax.grid(lw=0.3, alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=handles_legend, fontsize=7.5, loc="upper left",
                  framealpha=0.9, title="target", title_fontsize=7.5)
    fig.legend(
        handles=[
            plt.scatter([], [], s=65, facecolors="none", edgecolors="#555",
                        linewidths=1.8, label="offline"),
            plt.scatter([], [], s=65, color="#555", edgecolors="white",
                        linewidths=0.5, label="oracle"),
        ],
        fontsize=8.5, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, 1.03),
        title="○ offline (model-dep only) → ● oracle (+q/k sim)",
        title_fontsize=8.5, framealpha=0.9,
    )
    fig.suptitle(
        f"Predictability landscape — {title_suffix} | {cfg.label}",
        fontsize=12, fontweight="bold", y=1.07,
    )
    if out_dir is not None:
        _savefig(fig, out_dir / "r2_vs_lift.png")
    plt.show()
    return fig


def _build_pivot(df_results: pd.DataFrame, lift_col: str = "lift") -> Optional[pd.DataFrame]:
    pivot_rows: dict = {}
    for (tgt, var), row in df_results.iterrows():
        if tgt not in pivot_rows:
            pivot_rows[tgt] = {}
        pivot_rows[tgt][f"R2_{var}"]           = row.get("R2")
        pivot_rows[tgt][f"{lift_col}_{var}"]   = row.get(lift_col) or row.get("lift")
    pivot = pd.DataFrame(pivot_rows).T
    needed = {f"R2_offline", f"R2_oracle",
              f"{lift_col}_offline", f"{lift_col}_oracle"}
    return pivot if needed.issubset(pivot.columns) else None


def run_cross_prompt_experiment(
    df:             pd.DataFrame,
    cfg:            ModelConfig,
    target:         Optional[str] = None,
    prompt_sources: Optional[List[str]] = None,
    tune:           bool = True,
    n_trials:       int  = 10,
    seed:           int  = 42,
    out_dir:        Optional[Path] = None,
) -> dict:
    target  = target or cfg.default_target
    out_dir = Path(out_dir) / cfg.key / "cross_prompt" if out_dir else None

    if prompt_sources:
        df = df[df["prompt_source"].isin(prompt_sources)].copy()

    df_train, df_val, df_test = split_by_prompt(df, seed=seed)
    print(f"\n{'═'*62}")
    print(f"  {cfg.label} — Cross-Prompt  [target: {target}]")
    print(f"{'═'*62}")
    print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

    all_results: dict = {}
    metrics_rows: list = []

    for var_name, features in FEATURE_SETS.items():
        feats = [f for f in features if f in df.columns]
        print(f"\n  ── {var_name} ({len(feats)} features) ──")

        best_params = None
        if tune:
            print(f"     Optuna tuning ({n_trials} trials)...", end=" ", flush=True)
            best_params = tune_lgbm(df_train, df_val, feats, target, n_trials, seed)
            print("done.")

        model = train_lgbm(df_train, df_val, feats, target,
                           params_override=best_params, seed=seed)
        metrics, df_pred = evaluate(model, df_test, feats, target, cfg,
                                    df_train=df_train, baseline_fn="head_mean")

        print(f"     R²: {metrics['R2']:.4f}  MAE: {metrics['MAE']:.5f}", end="")
        if "MAE_baseline" in metrics:
            print(f"  baseline: {metrics['MAE_baseline']:.5f}"
                  f"  lift: {metrics.get('lift', 0):+.1%}", end="")
        print()

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            df_pred.to_parquet(
                out_dir / f"predictions_{_safe_name(target)}_{var_name}.parquet",
                index=False,
            )

        all_results[var_name] = {
            "model":       model,
            "metrics":     metrics,
            "predictions": df_pred,
            "features":    feats,
            "best_params": best_params,
        }
        metrics_rows.append({"variant": var_name, **metrics})

    if out_dir is not None:
        _save_csv(
            pd.DataFrame(metrics_rows).set_index("variant"),
            out_dir / f"metrics_{_safe_name(target)}.csv",
        )
        if "oracle" in all_results:
            plot_top_features(all_results, cfg, target, out_dir=out_dir)

    return all_results


def run_cross_head_experiment(
    df:            pd.DataFrame,
    cfg:           ModelConfig,
    target:        Optional[str] = None,
    prompt_source: Optional[str] = None,
    tune:          bool = True,
    n_trials:      int  = 10,
    seed:          int  = 42,
    out_dir:       Optional[Path] = None,
) -> dict:
    target  = target or cfg.default_target
    out_dir = Path(out_dir) / cfg.key / "cross_head" if out_dir else None

    if prompt_source:
        df = df[df["prompt_source"] == prompt_source].copy()

    print(f"\n{'═'*62}")
    print(f"  {cfg.label} — Cross-Head  [target: {target}]")
    print(f"{'═'*62}")

    results_ch:   dict  = {}
    metrics_rows: list  = []

    for var_name, feats in FEATURE_SETS.items():
        feats_avail = [f for f in feats if f in df.columns]
        print(f"\n  ── {var_name} ({len(feats_avail)} features) ──")

        df_agg, feat_cols = aggregate_per_head(df, feats_avail, target, cfg)
        df_tr, df_va, df_te = split_by_head(df_agg, seed=seed)
        print(f"     Heads — train: {len(df_tr)} | val: {len(df_va)} | test: {len(df_te)}")

        best_params = None
        if tune:
            print(f"     Optuna tuning ({n_trials} trials)...", end=" ", flush=True)
            best_params = tune_lgbm(df_tr, df_va, feat_cols,
                                    "target_median", n_trials, seed)
            print("done.")

        model, feats_used = train_lgbm_head(df_tr, df_va, feat_cols,
                                             params_override=best_params, seed=seed)
        metrics, df_pred  = evaluate_head(model, feats_used, df_tr, df_te,
                                          label=f"CH/{var_name}")

        print(f"     R²: {metrics['R2']:.4f}  MAE: {metrics['MAE']:.5f}"
              f"  lift_vs_NN: {metrics['lift_nn']:+.1%}")

        results_ch[var_name] = {
            "model":       model,
            "feats":       feats_used,
            "metrics":     metrics,
            "preds":       df_pred,
            "df_train":    df_tr,
            "best_params": best_params,
        }
        metrics_rows.append({"variant": var_name, **metrics})

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(
            pd.DataFrame(metrics_rows).set_index("variant"),
            out_dir / f"metrics_{_safe_name(target)}.csv",
        )
        plot_shap_crosshead(results_ch, cfg, target, out_dir=out_dir)

    return results_ch


def run_cross_prompt_all_targets(
    df:             pd.DataFrame,
    cfg:            ModelConfig,
    targets:        List[str] = ALL_TARGETS,
    prompt_sources: Optional[List[str]] = None,
    tune:           bool = True,
    n_trials:       int  = 10,
    seed:           int  = 42,
    out_dir:        Optional[Path] = None,
) -> pd.DataFrame:
    exp_tag = "cross_prompt"
    out_dir = Path(out_dir) / cfg.key / "all_targets" if out_dir else None

    if prompt_sources:
        df = df[df["prompt_source"].isin(prompt_sources)].copy()

    df_train, df_val, df_test = split_by_prompt(df, seed=seed)
    print(f"\n{'═'*62}")
    print(f"  {cfg.label} — All-Targets Cross-Prompt")
    print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")
    print(f"{'═'*62}\n")

    df_results = _run_all_targets(
        df_train, df_val, df_test, df,
        cfg, targets, tune, n_trials, seed,
        exp_tag=exp_tag, baseline_fn="head_mean",
        out_dir=out_dir,
    )

    if out_dir is not None and len(df_results):
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(df_results, out_dir / "all_targets_results.csv")
        pivot = _build_pivot(df_results, lift_col="lift")
        if pivot is not None:
            plot_r2_vs_lift_panels(pivot, cfg, out_dir=out_dir,
                                   title_suffix="Cross-Prompt", lift_col="lift")
    return df_results


def run_length_generalization(
    df:             pd.DataFrame,
    cfg:            ModelConfig,
    targets:        List[str] = ALL_TARGETS,
    train_lengths:  Tuple[int, ...] = (64, 128, 256),   # FIX: removed 0
    test_lengths:   Tuple[int, ...] = (512,),
    val_frac:       float = 0.15,
    tune:           bool  = True,
    n_trials:       int   = 10,
    seed:           int   = 42,
    out_dir:        Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generalisation to unseen prompt lengths.
    Baseline: length_corrected_head_mean (mean + OLS correction for log(prompt_len)).
    Lift = model improvement relative to this corrected baseline.
    """
    exp_tag = "length_gen"
    out_dir = Path(out_dir) / cfg.key / "length_generalization" if out_dir else None

    # Guard: dataset must contain prompt_len and the required lengths
    if "prompt_len" not in df.columns:
        print(f"  [SKIP] {cfg.label} — length_generalization: "
              f"column 'prompt_len' not found.")
        return pd.DataFrame()
    available = set(df["prompt_len"].unique())
    if not any(l in available for l in train_lengths) or \
       not any(l in available for l in test_lengths):
        print(f"  [SKIP] {cfg.label} — length_generalization: "
              f"required lengths not present. "
              f"Available: {sorted(available)}, "
              f"need train={train_lengths}, test={test_lengths}.")
        return pd.DataFrame()

    df_train, df_val, df_test = split_by_length_stratified(
        df,
        train_lengths=train_lengths,
        test_lengths=test_lengths,
        val_frac=val_frac,
        seed=seed,
    )
    print(f"\n{'═'*62}")
    print(f"  {cfg.label} — Length Generalisation")
    print(f"  Train lengths: {train_lengths}  →  Test lengths: {test_lengths}")
    print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")
    print(f"{'═'*62}\n")

    df_results = _run_all_targets(
        df_train, df_val, df_test, df,
        cfg, targets, tune, n_trials, seed,
        exp_tag=exp_tag, baseline_fn="length_corrected",
        out_dir=out_dir,
    )

    if out_dir is not None and len(df_results):
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(df_results, out_dir / "length_generalization_results.csv")
        pivot = _build_pivot(df_results, lift_col="lift")
        if pivot is not None:
            plot_r2_vs_lift_panels(pivot, cfg, out_dir=out_dir,
                                   title_suffix="Length Generalisation", lift_col="lift")
    return df_results



def run_cross_head_all_targets(
    df:       pd.DataFrame,
    cfg:      ModelConfig,
    targets:  List[str] = ALL_TARGETS,
    tune:     bool = True,
    n_trials: int  = 10,
    seed:     int  = 42,
    out_dir:  Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run cross-head experiment on every target in `targets`.
    For each target: aggregate per head, split by head, train LightGBM,
    evaluate with nearest-neighbour baseline.
    Saves metrics CSV and r2_vs_lift plot under <out_dir>/<model_key>/cross_head/.
    Use run_cross_head_experiment() for interactive single-target exploration.
    """
    out_dir_exp = Path(out_dir) / cfg.key / "cross_head" if out_dir else None

    print(f"\n{'═'*62}")
    print(f"  {cfg.label} — Cross-Head (all targets)")
    print(f"{'═'*62}\n")

    rows: list = []
    valid_targets = [t for t in targets if t in df.columns]
    total = len(valid_targets) * len(FEATURE_SETS)
    done  = 0

    for target in valid_targets:
        best_params_per_variant: Dict[str, Optional[dict]] = {v: None for v in FEATURE_SETS}

        # Tuning: done once per (target, variant) on the aggregated dataset
        if tune:
            for var_name, feats in FEATURE_SETS.items():
                feats_avail = [f for f in feats if f in df.columns]
                df_agg, feat_cols = aggregate_per_head(df, feats_avail, target, cfg)
                df_tr, df_va, _   = split_by_head(df_agg, seed=seed)
                if len(df_tr) < 5:
                    continue
                print(f"     Optuna [CH/{var_name}] {target} ({n_trials} trials)...",
                      end=" ", flush=True)
                best_params_per_variant[var_name] = tune_lgbm(
                    df_tr, df_va, feat_cols, "target_median", n_trials, seed)
                print("done.")

            if out_dir_exp is not None:
                import json
                params_path = out_dir_exp / f"best_params_{_safe_name(target)}.json"
                params_path.parent.mkdir(parents=True, exist_ok=True)
                with open(params_path, "w") as fp:
                    json.dump(best_params_per_variant, fp, indent=2)

        for var_name, feats in FEATURE_SETS.items():
            done += 1
            feats_avail = [f for f in feats if f in df.columns]
            print(f"[{done:>3}/{total}] {target:<38} {var_name}", end="  ")

            df_agg, feat_cols = aggregate_per_head(df, feats_avail, target, cfg)
            df_tr, df_va, df_te = split_by_head(df_agg, seed=seed)

            if len(df_tr) < 5 or len(df_te) < 2:
                print("SKIP"); continue

            model, feats_used = train_lgbm_head(
                df_tr, df_va, feat_cols,
                params_override=best_params_per_variant.get(var_name),
                seed=seed,
            )
            metrics, df_pred = evaluate_head(
                model, feats_used, df_tr, df_te,
                label=f"CH/{var_name}/{target}",
            )
            row = {
                "target":  target,
                "variant": var_name,
                **{k: round(v, 5) if isinstance(v, float) else v
                   for k, v in metrics.items()
                   if k not in ("label",)},
            }
            rows.append(row)
            print(f"R²={metrics['R2']:.4f}  MAE={metrics['MAE']:.5f}  "
                  f"lift_nn={metrics['lift_nn']:+.1%}")

    if not rows:
        return pd.DataFrame(columns=["target", "variant"]).set_index(["target", "variant"])

    df_results = (
        pd.DataFrame(rows)
        .set_index(["target", "variant"])
        .sort_index()
    )

    if out_dir_exp is not None and len(df_results):
        out_dir_exp.mkdir(parents=True, exist_ok=True)
        _save_csv(df_results, out_dir_exp / "all_targets_results.csv")
        pivot = _build_pivot(df_results, lift_col="lift_nn")
        if pivot is not None:
            plot_r2_vs_lift_panels(pivot, cfg, out_dir=out_dir_exp,
                                   title_suffix="Cross-Head", lift_col="lift_nn")

    return df_results


def run_all_models(
    datasets:    Dict[str, pd.DataFrame],
    model_cfgs:  List[ModelConfig],
    experiments: List[str] = ["cross_prompt", "cross_head", "all_targets",
                               "length_generalization"],
    tune:        bool  = True,
    n_trials:    int   = 10,
    seed:        int   = 42,
    out_dir:     Path  = Path("results"),
) -> Dict[str, dict]:
    """
    Run all experiments for every model in model_cfgs.

    Parameters
    ----------
    datasets     : {model_key: dataframe}
                   For length_generalization must contain prompts at
                   multiple lengths (column prompt_len).
    model_cfgs   : list of ModelConfig
    experiments  : subset of
                   ["cross_prompt", "cross_head", "all_targets",
                    "length_generalization"]
                   Default: all four.
    tune         : enable Optuna (True recommended)
    n_trials     : Optuna trials per variant × target
    out_dir      : root results directory

    Returns
    -------
    {model_key: {experiment_name: results_df_or_dict}}
    """
    out_dir = Path(out_dir)
    all_results: dict = {}

    for cfg in model_cfgs:
        if cfg.key not in datasets:
            print(f"[SKIP] {cfg.label}: no dataset provided.")
            continue
        df = datasets[cfg.key]
        model_results: dict = {}

        if "cross_prompt" in experiments:
            # Iterates over ALL_TARGETS with cross-prompt split.
            # Saves under <out_dir>/<model>/cross_prompt/.
            # Use run_cross_prompt_experiment() for interactive single-target.
            model_results["cross_prompt"] = run_cross_prompt_all_targets(
                df, cfg, tune=tune, n_trials=n_trials, seed=seed,
                out_dir=out_dir,
            )

        if "cross_head" in experiments:
            # Iterates over ALL_TARGETS with cross-head split.
            # Use run_cross_head_experiment() for interactive single-target.
            model_results["cross_head"] = run_cross_head_all_targets(
                df, cfg, tune=tune, n_trials=n_trials, seed=seed, out_dir=out_dir,
            )

        if "all_targets" in experiments:
            model_results["all_targets"] = run_cross_prompt_all_targets(
                df, cfg, tune=tune, n_trials=n_trials, seed=seed, out_dir=out_dir,
            )

        if "length_generalization" in experiments:
            model_results["length_generalization"] = run_length_generalization(
                df, cfg, tune=tune, n_trials=n_trials, seed=seed, out_dir=out_dir,
            )

        all_results[cfg.key] = model_results

    return all_results
