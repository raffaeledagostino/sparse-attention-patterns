"""Centralized project configuration constants."""

from pathlib import Path


# ── Model ────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B"
MODEL_NAME_SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME_SMALL_2 = "Qwen/Qwen3-0.6B"

TRUST_REMOTE_CODE = True
LOCAL_FILES_ONLY = False


# ── Device ───────────────────────────────────────────────────────────────
DEVICE_MPS = "mps"
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"
DEVICE_MAP = "auto"


# ── Attention ────────────────────────────────────────────────────────────
ATTN_IMPLEMENTATION = "eager"
INFERENCE_FP16_CUDA = True


# ── Dataset / Pipeline ───────────────────────────────────────────────────
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "train"
DATASET_TEXT_COLUMN = "text"
MIN_CHARS = 100
TARGET_TOKENS = 64
RANDOM_SEED = 42
PROMPT_PREVIEW_CHARS = 100


# ── Persistence ──────────────────────────────────────────────────────────
OUTPUT_PATH = Path("data/Qwen3_4B_512tok.parquet")
PRIMARY_KEY = ["model_name", "prompt_id", "layer_idx", "head_idx"]
# trigger test
