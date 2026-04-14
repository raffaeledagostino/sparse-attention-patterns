"""
Batch analysis orchestration for processing multiple prompts with checkpoint-aware resumption.

This module handles batch processing of multiple prompts with:
- Checkpoint-aware resumption (skips already-analyzed prompts)
- Progress tracking and elapsed time reporting
- Device memory cleanup between prompts
- Atomic persistence with deduplication

Status: Under development. Use core.analyzer and pipeline.run_analysis() for stable single-prompt analysis.
"""

import gc
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch

from config import OUTPUT_PATH, PRIMARY_KEY
from data.persistence import save_results


def _load_existing_prompt_ids(output_path: Path) -> set[str]:
    """
    Load existing prompt IDs without reading the full parquet payload.
    
    Useful for checkpoint-aware batch processing: determines which prompts
    have already been analyzed and should be skipped.
    
    Args:
        output_path: Path to the Parquet file.
    
    Returns:
        Set of prompt_id strings already in the dataset (or empty set if file doesn't exist).
    """
    if not output_path.exists():
        return set()

    prompt_ids = pd.read_parquet(output_path, columns=["prompt_id"])["prompt_id"]
    return set(prompt_ids.astype(str).tolist())


def _format_elapsed(seconds: float) -> str:
    """
    Format elapsed seconds as a human-readable string (e.g., "1m 23s").
    
    Args:
        seconds: Elapsed time in seconds.
    
    Returns:
        Formatted string like "1m 23s".
    """
    total_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{minutes}m {remaining_seconds:02d}s"


def _cleanup_device(device: str) -> None:
    """
    Release prompt-level memory before moving to the next item.
    
    This is part of the Eager Eviction pattern: ensures that cached memory
    on the device (MPS, CUDA) is truly freed, not just marked for reclamation.
    
    Args:
        device: The computation device ("mps", "cuda", or "cpu").
    """
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def run_analysis_batch(
    analyzer,
    source,
    prompt_indices: Iterable[int],
    layer_indices=None,
    head_indices=None,
    output_path: Path = OUTPUT_PATH,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run checkpoint-aware batch analysis for a sequence of prompts.
    
    This function processes multiple prompts with the following features:
    - Checkpoint resumption: skips prompts already in output_path
    - Progress tracking: prints elapsed time and per-prompt status
    - Device memory cleanup: calls _cleanup_device() between prompts
    - Atomic persistence: uses save_results() for safe append/dedup
    
    Args:
        analyzer: LightweightAttentionAnalyzer instance.
        source: Prompt source object with get_prompt(idx) and source_tag, target_tokens attrs.
        prompt_indices: Iterable of prompt indices to process.
        layer_indices: If provided, only analyze these layer indices.
        head_indices: If provided, only analyze these head indices.
        output_path: Path to Parquet file for persistence (default: config.OUTPUT_PATH).
        device: Device in use ("mps", "cuda", "cpu") for cleanup calls.
    
    Returns:
        DataFrame with all results (existing + newly added) from output_path.
        If output_path doesn't exist, returns the latest results DataFrame.
    
    Raises:
        Exception: Propagates any errors from analyzer.analyze_prompt() or save_results().
    """
    prompt_indices = list(prompt_indices)
    total = len(prompt_indices)
    loop_start = time.perf_counter()
    latest_df = pd.DataFrame()
    existing_prompt_ids = _load_existing_prompt_ids(output_path)

    for index, prompt_idx in enumerate(prompt_indices):
        prompt, prompt_id = source.get_prompt(prompt_idx)

        # Skip prompts that have already been analyzed
        if prompt_id in existing_prompt_ids:
            elapsed = _format_elapsed(time.perf_counter() - loop_start)
            print(f"[Skip] {prompt_id} already done")
            print(
                f"[Progress] {index + 1}/{total} | elapsed: {elapsed} | last: {prompt_id}"
            )
            continue

        # Analyze the prompt
        results = analyzer.analyze_prompt(
            prompt,
            max_length=source.target_tokens,
            layer_indices=layer_indices,
            head_indices=head_indices,
        )

        # Attach metadata
        for result in results:
            result["prompt_id"] = prompt_id
            result["prompt_source"] = source.source_tag

        # Persist results
        latest_df = save_results(results, output_path, PRIMARY_KEY)
        existing_prompt_ids.add(prompt_id)

        # Progress reporting
        elapsed = _format_elapsed(time.perf_counter() - loop_start)
        print(f"[Done] {prompt_id} saved ({index + 1}/{total})")
        print(
            f"[Progress] {index + 1}/{total} | elapsed: {elapsed} | last: {prompt_id}"
        )

        # Cleanup device memory before next prompt
        _cleanup_device(device)

    # Return final dataset (load from disk if Parquet exists)
    if output_path.exists():
        return pd.read_parquet(output_path)
    return latest_df
