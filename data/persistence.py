"""Persistence helpers for analysis outputs."""

from pathlib import Path

import pandas as pd


def save_results(results: list, output_path: Path, primary_key: list) -> pd.DataFrame:
    """Save records to parquet with idempotent merge-on-primary-key behavior."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(results)

    if output_path.exists():
        df_existing = pd.read_parquet(output_path)
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

    df_final.to_parquet(output_path, index=False)
    print(f"[Save] rows_added={rows_added} total_rows={len(df_final)}")
    return df_final
