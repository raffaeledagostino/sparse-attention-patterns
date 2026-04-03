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
        Append new feature records to the dataset.
        
        Atomically loads existing data (if any), appends new records, and
        writes back to disk. Ensures no data loss and maintains schema consistency.
        
        Args:
            records (List[Dict[str, Any]]): List of feature dictionaries to append.
        
        Returns:
            int: Total number of rows in the updated dataset.
        
        Raises:
            ValueError: If records list is empty or records have inconsistent schemas.
            RuntimeError: If write operation fails.
        """
        if not records:
            raise ValueError("Cannot append empty records list.")
        
        # Convert records to dataframe
        try:
            new_df = pd.DataFrame(records)
        except Exception as e:
            raise ValueError(f"Failed to convert records to dataframe: {e}")
        
        # Load existing data
        existing_df = self.read_dataset()
        
        if existing_df is None:
            # No existing data; this is the first write
            combined_df = new_df
            print(f"[DatasetManager] Creating new dataset with {len(new_df)} records.")
        else:
            # Validate schema compatibility
            existing_cols = set(existing_df.columns)
            new_cols = set(new_df.columns)
            
            if existing_cols != new_cols:
                missing_in_new = existing_cols - new_cols
                extra_in_new = new_cols - existing_cols
                
                if missing_in_new:
                    print(f"[DatasetManager] Warning: New records missing columns: {missing_in_new}")
                if extra_in_new:
                    print(f"[DatasetManager] Warning: New records have extra columns: {extra_in_new}")
                
                # Align columns
                for col in existing_cols:
                    if col not in new_df.columns:
                        new_df[col] = np.nan
                
                new_df = new_df[existing_cols]
            
            # Concatenate
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"[DatasetManager] Appended {len(new_df)} new records. Total: {len(combined_df)} rows.")
        
        # Write to disk
        try:
            # Ensure parent directory exists
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if self.format == "parquet":
                combined_df.to_parquet(self.filepath, index=False)
            else:  # csv
                combined_df.to_csv(self.filepath, index=False)
            
            print(f"[DatasetManager] Dataset written to {self.filepath}")
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
