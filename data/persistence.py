"""Persistence helpers for analysis outputs."""

from pathlib import Path

import pandas as pd


def _strip_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop persisted index helper columns so they can be rebuilt deterministically."""
    cols_to_drop = [col for col in ["run_idx", "sub_idx"] if col in df.columns]
    if cols_to_drop:
        return df.drop(columns=cols_to_drop)
    return df


def _apply_hierarchical_index(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a two-level index: run_idx -> sub_idx (head-layer row within run)."""
    if df.empty:
        empty = df.copy()
        empty["run_idx"] = pd.Series(dtype="object")
        empty["sub_idx"] = pd.Series(dtype="int64")
        return empty.set_index(["run_idx", "sub_idx"], drop=True)

    indexed = df.copy()

    if "prompt_id" in indexed.columns and "model_name" in indexed.columns:
        indexed["run_idx"] = (
            indexed["model_name"].astype(str) + "::" + indexed["prompt_id"].astype(str)
        )
    elif "prompt_id" in indexed.columns:
        indexed["run_idx"] = indexed["prompt_id"].astype(str)
    else:
        indexed["run_idx"] = [f"run_{i}" for i in range(len(indexed))]

    sort_cols = ["run_idx"] + [
        col for col in ["layer_idx", "head_idx"] if col in indexed.columns
    ]
    indexed = indexed.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    indexed["sub_idx"] = indexed.groupby("run_idx").cumcount()
    indexed = indexed.set_index(["run_idx", "sub_idx"], drop=True)
    indexed.index.names = ["run_idx", "sub_idx"]
    return indexed


def save_results(results: list, output_path: Path, primary_key: list) -> pd.DataFrame:
    """Save records to parquet with idempotent merge-on-primary-key behavior."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = _strip_index_columns(pd.DataFrame(results))

    if output_path.exists():
        df_existing = _strip_index_columns(pd.read_parquet(output_path).reset_index(drop=True))
        n_before = len(df_existing)
        df_final = (
            pd.concat([df_existing, df_new], ignore_index=True)
            .drop_duplicates(subset=primary_key, keep="last")
            .reset_index(drop=True)
        )
        rows_added = len(df_final) - n_before
    else:
        df_final = df_new
        rows_added = len(df_new)

    df_final = _apply_hierarchical_index(df_final)
    df_final.to_parquet(output_path, index=True)
    print(f"[Save] rows_added={rows_added} total_rows={len(df_final)}")
    return df_final
