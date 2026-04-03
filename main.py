"""
Entry point for the sparse attention analysis pipeline.

This script orchestrates the complete workflow:
  1. Load configuration and command-line arguments.
  2. Initialize the LightweightAttentionAnalyzer.
  3. Process one or more prompts.
  4. Extract features from attention heads.
  5. Append results to a persistent Parquet/CSV dataset.

Usage:
    python main.py --prompt "Your prompt here" [--model MODEL_NAME] [--output OUTPUT_FILE]
    
    Example:
        python main.py --prompt "Hello, how are you?" --model Qwen/Qwen2.5-0.5B-Instruct \\
            --output features.parquet --layers 0 1 2 --heads 0 1 2
"""

import argparse
import sys
from typing import List, Optional
from pathlib import Path

from core.analyzer import LightweightAttentionAnalyzer
from core.dataset_manager import DatasetManager


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract mathematical features from LLM attention mechanisms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Core arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt to analyze.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model identifier (default: Qwen/Qwen2.5-0.5B-Instruct).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="features.parquet",
        help="Output dataset file path (default: features.parquet).",
    )
    
    # Analysis configuration
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum token sequence length (default: 128).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        help="Specific layer indices to analyze (default: all layers).",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="*",
        help="Specific head indices to analyze (default: all heads).",
    )
    
    # Device and performance
    parser.add_argument(
        "--device",
        type=str,
        choices=["mps", "cuda", "cpu"],
        help="Computation device. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached models (don't download).",
    )
    
    # Dataset management
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format: 'parquet' or 'csv' (default: parquet).",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset info and exit (don't run analysis).",
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point.
    
    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    args = parse_arguments()
    
    print("=" * 80)
    print("Sparse Attention Analysis Pipeline")
    print("=" * 80)
    
    # Initialize dataset manager
    try:
        dataset_manager = DatasetManager(args.output, format=args.format)
    except Exception as e:
        print(f"[ERROR] Failed to initialize dataset manager: {e}")
        return 1
    
    # Handle --info flag
    if args.info:
        info = dataset_manager.get_dataset_info()
        if info is None:
            print(f"[INFO] No dataset found at {args.output}")
        else:
            print(f"[INFO] Dataset info:")
            print(f"  Shape: {info['shape']}")
            print(f"  Columns: {info['columns']}")
            print(f"  Null counts: {info['null_counts']}")
        return 0
    
    # Initialize analyzer
    try:
        analyzer = LightweightAttentionAnalyzer(
            model_name=args.model,
            device=args.device,
            local_files_only=args.local_files_only,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize analyzer: {e}")
        return 1
    
    # Analyze prompt
    try:
        results = analyzer.analyze_prompt(
            prompt=args.prompt,
            max_length=args.max_length,
            layer_indices=args.layers if args.layers else None,
            head_indices=args.heads if args.heads else None,
        )
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        return 1
    
    if not results:
        print("[WARNING] No results were generated.")
        return 1
    
    # Append to dataset
    try:
        total_rows = dataset_manager.append_records(results)
        print(f"\n[SUCCESS] Analysis complete. Dataset now has {total_rows} total rows.")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        return 1
    
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
