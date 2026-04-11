"""
Dataset management utilities for appending feature data safely.

This module handles reading, validating, and appending feature records to
persistent storage (Parquet or CSV format) without overwriting existing data.

The design prioritizes safety: existing data is always preserved, and new
records are appended atomically.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np

from config import PRIMARY_KEY

PRIMARY_KEYS = PRIMARY_KEY


class DatasetManager:
    """
    Manages persistent storage of attention feature datasets.
    
    Handles reading, validating, and appending feature records to Parquet
    or CSV files. Ensures data integrity through atomic operations and
    schema consistency checks.
    
    Attributes:
        filepath (Path): Path to the dataset file.
        format (str): "parquet" or "csv" format identifier.
    """
    
    def __init__(self, filepath: str, format: str = "parquet"):
        """
        Initialize the dataset manager.
        
        Args:
            filepath (str): Path to the dataset file.
            format (str): File format - "parquet" or "csv" (default: "parquet").
        
        Raises:
            ValueError: If format is not "parquet" or "csv".
        """
        if format not in ("parquet", "csv"):
            raise ValueError(f"Unsupported format: {format}. Must be 'parquet' or 'csv'.")
        
        self.filepath = Path(filepath)
        self.format = format
    
    def read_dataset(self) -> Optional[pd.DataFrame]:
        """
        Read existing dataset from disk.
        
        Returns:
            Optional[pd.DataFrame]: Loaded dataframe if file exists, None otherwise.
        
        Raises:
            RuntimeError: If file exists but cannot be read.
        """
        if not self.filepath.exists():
            return None
        
        try:
            if self.format == "parquet":
                df = pd.read_parquet(self.filepath)
            else:  # csv
                df = pd.read_csv(self.filepath)
            
            print(f"[DatasetManager] Loaded existing dataset: {len(df)} rows.")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to read dataset from {self.filepath}: {e}")
    
def append_records(self, records: List[Dict[str, Any]]) -> int:
    """
    Incrementally update the dataset with new records.

    Three cases are handled transparently:
      1. File does not exist -> create from scratch
      2. New (model, prompt_id, layer, head) combos -> append new rows
      3. Same combos, new columns  -> merge new features into existing rows
         (existing rows get NaN for new columns until reprocessed)

    Args:
        records: List of feature dicts, each must contain the four PRIMARY_KEYS.

    Returns:
        int: Total number of rows in the updated dataset.
    """
    if not records:
        raise ValueError("Cannot append empty records list.")

    try:
        df_new = pd.DataFrame(records)
    except Exception as e:
        raise ValueError(f"Failed to convert records to dataframe: {e}")

    # Validate PRIMARY_KEYS presence
    missing_keys = [k for k in PRIMARY_KEYS if k not in df_new.columns]
    if missing_keys:
        raise ValueError(f"Records missing primary key columns: {missing_keys}")

    # --- Case 1: no existing dataset ---
    existing_df = self.read_dataset()
    if existing_df is None:
        combined_df = df_new
        print(f"[DatasetManager] Creating new dataset: {len(df_new)} rows, {len(df_new.columns)} cols.")

    else:
        # Build key sets for comparison
        def _key_set(df: pd.DataFrame):
            return set(zip(*[df[k].astype(str) for k in PRIMARY_KEYS]))

        old_keys = _key_set(existing_df)
        new_keys = _key_set(df_new)

        truly_new = new_keys - old_keys    # combos not yet in dataset
        overlap   = new_keys & old_keys    # combos already present

        combined_df = existing_df.copy()

        # --- Case 2: append genuinely new rows ---
        if truly_new:
            mask = df_new.apply(
                lambda r: tuple(str(r[k]) for k in PRIMARY_KEYS) in truly_new,
                axis=1
            )
            df_append = df_new[mask].copy()
            # Fill missing columns from existing schema with NaN
            for col in existing_df.columns:
                if col not in df_append.columns:
                    df_append[col] = np.nan
            combined_df = pd.concat([combined_df, df_append], ignore_index=True)
            print(f"[DatasetManager] Appended {len(df_append)} new rows.")

        # --- Case 3: new feature columns for existing rows ---
        new_feature_cols = [c for c in df_new.columns if c not in existing_df.columns]
        if new_feature_cols and overlap:
            cols_to_merge = PRIMARY_KEYS + new_feature_cols
            # Cast keys to str for merge safety
            merge_right = df_new[cols_to_merge].copy()
            for k in PRIMARY_KEYS:
                combined_df[k] = combined_df[k].astype(str)
                merge_right[k] = merge_right[k].astype(str)
            combined_df = pd.merge(combined_df, merge_right, on=PRIMARY_KEYS, how="left")
            print(f"[DatasetManager] Added {len(new_feature_cols)} new feature columns: {new_feature_cols}")

        if not truly_new and not new_feature_cols:
            print("[DatasetManager] Warning: no new rows and no new columns detected. Nothing written.")
            return len(combined_df)

    # Write atomically
    try:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if self.format == "parquet":
            combined_df.to_parquet(self.filepath, index=False)
        else:
            combined_df.to_csv(self.filepath, index=False)
        print(f"[DatasetManager] Dataset written to {self.filepath}: "
              f"{len(combined_df)} rows × {len(combined_df.columns)} cols.")
    except Exception as e:
        raise RuntimeError(f"Failed to write dataset to {self.filepath}: {e}")

    return len(combined_df)
    
    def get_dataset_info(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve summary information about the dataset.
        
        Returns:
            Optional[Dict[str, Any]]: Dictionary with shape, columns, and stats,
                                      or None if dataset doesn't exist.
        """
        df = self.read_dataset()
        if df is None:
            return None
        
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
        }
    
    def delete_dataset(self) -> bool:
        """
        Delete the dataset file from disk.
        
        Returns:
            bool: True if file was deleted, False if it didn't exist.
        
        Raises:
            RuntimeError: If deletion fails.
        """
        if not self.filepath.exists():
            return False
        
        try:
            self.filepath.unlink()
            print(f"[DatasetManager] Deleted {self.filepath}")
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete {self.filepath}: {e}")
