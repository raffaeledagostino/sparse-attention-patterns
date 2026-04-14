"""
Single-prompt analysis pipeline orchestration.

This module handles the core single-prompt analysis workflow:
- Tokenization and forward pass
- Feature extraction per attention head
- Persistence with deduplication

For batch processing of multiple prompts, see batch_analysis.py.
"""

from pathlib import Path

import pandas as pd

from config import OUTPUT_PATH, PRIMARY_KEY
from data.persistence import save_results

# For backwards compatibility: batch functions have moved to batch_analysis.py
# Import them here if needed:
# from batch_analysis import run_analysis_batch


def run_analysis(
    analyzer,
    source,
    prompt_idx: int = 0,
    layer_indices=None,
    head_indices=None,
    output_path=OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Run a complete analysis pass for a single prompt and persist the result dataset.
    
    This is the core single-prompt analysis function. It:
    1. Fetches a prompt from the source
    2. Runs the analyzer on that prompt
    3. Attaches metadata (prompt_id, prompt_source)
    4. Saves results with deduplication
    
    Args:
        analyzer: LightweightAttentionAnalyzer instance.
        source: Prompt source object with:
                - get_prompt(idx) -> (prompt_text, prompt_id)
                - source_tag: str (e.g., "wikiptext")
                - target_tokens: int (max tokens to process)
        prompt_idx: Index of the prompt to fetch from source (default: 0).
        layer_indices: If provided, only analyze these layer indices.
        head_indices: If provided, only analyze these head indices.
        output_path: Path to Parquet file for persistence (default: config.OUTPUT_PATH).
    
    Returns:
        DataFrame with persisted results (merged with any pre-existing data).
    
    Raises:
        Exception: Propagates any errors from analyzer or persistence layer.
    
    Example:
        >>> analyzer = LightweightAttentionAnalyzer("Qwen/Qwen2.5-0.5B-Instruct")
        >>> df = run_analysis(analyzer, source, prompt_idx=42)
        >>> print(len(df), "total rows in dataset")
    """
    # Fetch the prompt
    prompt, prompt_id = source.get_prompt(prompt_idx)
    
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

    # Persist and return
    return save_results(results, output_path, PRIMARY_KEY)
