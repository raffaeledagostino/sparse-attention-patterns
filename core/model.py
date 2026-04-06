"""
Model and tokenizer loading utilities.
"""

import inspect
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    _autocast_params = inspect.signature(torch.is_autocast_enabled).parameters
except ValueError:
    _autocast_params = ()

if len(_autocast_params) == 0:
    _orig_is_autocast_enabled = torch.is_autocast_enabled

    def _compat_is_autocast_enabled(device_type=None):
        return _orig_is_autocast_enabled()

    torch.is_autocast_enabled = _compat_is_autocast_enabled

from config import MODEL_NAME, ATTN_IMPLEMENTATION


def load_tokenizer(model_name: str = MODEL_NAME, **kwargs):
    """Load the tokenizer for the given model."""
    return AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, **kwargs
    )


def load_model(model_name: str = MODEL_NAME, quantize_8bit: bool = False,
               device_map: str = "auto", **kwargs):
    """
    Load a causal LM with attention and hidden-state outputs enabled.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    quantize_8bit : bool
        If True, load with bitsandbytes 8-bit quantization.
    device_map : str
        Device placement strategy (default: "auto").

    Returns
    -------
    model : AutoModelForCausalLM
    """
    extra = {}
    if quantize_8bit:
        extra["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation=ATTN_IMPLEMENTATION,
        **extra,
        **kwargs,
    )

    model.config.output_attentions = True
    model.config.output_hidden_states = True

    return model


def print_model_config(model):
    """Print key architectural parameters."""
    cfg = model.config
    print("=== Model Configuration ===")
    print(f"Number of layers:          {cfg.num_hidden_layers}")
    print(f"Attention heads (Q):       {cfg.num_attention_heads}")
    print(f"Key-Value heads:           {cfg.num_key_value_heads}")
    print(f"Hidden size:               {cfg.hidden_size}")
    print(f"Intermediate size (FFN):   {cfg.intermediate_size}")
    print(f"Vocab size:                {cfg.vocab_size}")
    print(f"Max position embeddings:   {cfg.max_position_embeddings}")
    if hasattr(cfg, "head_dim"):
        print(f"Head dim:                  {cfg.head_dim}")
