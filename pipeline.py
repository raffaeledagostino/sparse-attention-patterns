"""Top-level analysis pipeline orchestration."""

import pandas as pd

from config import OUTPUT_PATH, PRIMARY_KEY
from data.persistence import save_results


def run_analysis(
    analyzer,
    source,
    prompt_idx: int = 0,
    layer_indices=None,
    head_indices=None,
    output_path=OUTPUT_PATH,
) -> pd.DataFrame:
    """Run a full analysis pass for one prompt and persist the result dataset."""
    prompt, prompt_id = source.get_prompt(prompt_idx)
    results = analyzer.analyze_prompt(
        prompt,
        max_length=source.target_tokens,
        layer_indices=layer_indices,
        head_indices=head_indices,
    )

    for result in results:
        result["prompt_id"] = prompt_id
        result["prompt_source"] = source.source_tag

    return save_results(results, output_path, PRIMARY_KEY)
