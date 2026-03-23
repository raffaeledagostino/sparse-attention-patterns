"""
Central configuration for the attention-patterns project.
All experiment-wide constants and defaults live here.
"""

# ── Model ─────────────────────────────────────────────────────────────────
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
