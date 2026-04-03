"""
Core module for sparse attention analysis pipeline.

This package provides the fundamental building blocks for extracting
mathematical features from LLM attention mechanisms with memory-efficient
in-memory processing on Apple Silicon.
"""

from core.context import HeadContext
from core.features_library import FEATURE_REGISTRY, compute_diagonal_mass, compute_effective_rank_Q
from core.analyzer import LightweightAttentionAnalyzer
from core.dataset_manager import DatasetManager

__all__ = [
    "HeadContext",
    "FEATURE_REGISTRY",
    "compute_diagonal_mass",
    "compute_effective_rank_Q",
    "LightweightAttentionAnalyzer",
    "DatasetManager",
]
