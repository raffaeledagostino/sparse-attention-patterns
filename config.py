"""
Central configuration for the attention-patterns project.
All experiment-wide constants and defaults live here.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ── Model ─────────────────────────────────────────────────────────────────
#MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME = "Qwen/Qwen3-4B"
#MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ATTN_IMPLEMENTATION = "eager"  # required to get full attention matrices

# ── Architecture (auto-detected at runtime, but useful as reference) ─────
HEAD_DIM = 128
NUM_HEADS = 32          # num_attention_heads  (Q heads)
NUM_KV_HEADS = 8        # num_key_value_heads  (K/V heads, GQA)
GROUP_SIZE = NUM_HEADS // NUM_KV_HEADS  # 4

# ── Data ──────────────────────────────────────────────────────────────────
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DEFAULT_SEQ_LEN = 64
DEFAULT_START_IDX = 100

# ── Experiments ───────────────────────────────────────────────────────────
DEFAULT_NUM_PROMPTS = 50
RANDOM_SEED = 42

# ── Visualization ─────────────────────────────────────────────────────────
DEFAULT_CMAP = "viridis"
PRESOFTMAX_CMAP = "magma"
MASK_THRESHOLD = -1e6     # threshold for masking causal -inf in pre-softmax

# ── Similarity ────────────────────────────────────────────────────────────
BINARIZATION_THRESHOLD = 0.05

# ── Output ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
IMG_DIR = f"{OUTPUT_DIR}/img"
EXPLORATION_IMG_DIR = f"{IMG_DIR}/exploration"
SIMILARITY_IMG_DIR = f"{IMG_DIR}/similarity"
PCA_IMG_DIR = f"{IMG_DIR}/pca"
DISTRIBUTION_IMG_DIR = f"{IMG_DIR}/distribution"
CLUSTERING_IMG_DIR = f"{IMG_DIR}/clustering"


# ── Modular Extraction + Metrics Pipeline ─────────────────────────────────
@dataclass(frozen=True)
class SweepConfig:
	"""Layer/head sweep configuration for extraction and dataset building."""

	layers: list[int] = field(default_factory=lambda: [0])
	heads: list[int] = field(default_factory=lambda: [0])


@dataclass(frozen=True)
class CacheConfig:
	"""On-disk cache locations used to decouple inference and metric computation."""

	root_dir: Path = Path("outputs/cache")
	raw_dirname: str = "raw_tensors"
	prompts_dirname: str = "prompts"
	static_filename: str = "static_tensors.pt"
	metadata_filename: str = "metadata.pt"
	dataset_filename: str = "attention_features.parquet"

	@property
	def raw_dir(self) -> Path:
		return self.root_dir / self.raw_dirname

	@property
	def prompts_dir(self) -> Path:
		return self.raw_dir / self.prompts_dirname

	@property
	def static_path(self) -> Path:
		return self.raw_dir / self.static_filename

	@property
	def metadata_path(self) -> Path:
		return self.raw_dir / self.metadata_filename

	@property
	def dataset_path(self) -> Path:
		return self.root_dir / self.dataset_filename


@dataclass(frozen=True)
class PromptConfig:
	"""Prompt extraction strategy for standardized prompt batches."""

	n_prompts: int = 1
	seq_len: int = 32
	stride: int = 600
	start_idx: int = 1000


@dataclass(frozen=True)
class PipelineConfig:
	"""Top-level pipeline configuration for extraction and metrics."""

	model_name: str = MODEL_NAME
	attn_implementation: str = ATTN_IMPLEMENTATION
	prompt: PromptConfig = field(default_factory=PromptConfig)
	sweep: SweepConfig = field(default_factory=lambda: SweepConfig(layers=[0], heads=[0]))
	cache: CacheConfig = field(default_factory=CacheConfig)
	diag_deltas: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
	sink_excluded_keys: int = 3
	r95_threshold: float = 0.95


DEFAULT_PIPELINE_CONFIG = PipelineConfig()
