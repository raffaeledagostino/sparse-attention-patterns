# Sparse Attention Analysis Pipeline

A highly modular, memory-efficient PyTorch pipeline for extracting mathematical features from LLM attention matrices. Designed for Apple Silicon (M2) and single-machine inference constraints.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Basic Command](#basic-command)
  - [Command-Line Arguments](#command-line-arguments)
  - [Example Workflows](#example-workflows)
- [How to Add a New Feature](#how-to-add-a-new-feature)
- [How to Add a New Prompt](#how-to-add-a-new-prompt)
- [How to Add a New Model](#how-to-add-a-new-model)
- [Module Reference](#module-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline extracts mathematical features from attention heads in transformers (like Qwen, LLaMA, Mistral) and saves them to a tabular dataset (Parquet/CSV). It targets the constraints of single-machine inference on Apple Silicon hardware.

### Key Capabilities

- **In-Memory Analysis**: Forward pass computation fully in memory without intermediate disk writes.
- **Memory-Efficient**: Implements "Eager Eviction"—explicit tensor deletion after use and garbage collection after each layer to prevent OOM.
- **Modular Feature System**: Add new mathematical metrics without retraining or re-running all features.
- **Multi-Head Analysis**: Extracts features per attention head across specified layers.
- **GQA Support**: Correctly handles Grouped Query Attention (KV head replication).
- **Safe Dataset Append**: Never overwrites existing data; appends atomically to persistent storage.

### Mathematical Features Included

| Feature | Description |
|---------|-------------|
| `diagonal_mass_5` | Fraction of attention mass within diagonal band (width=5) |
| `q_sim_consecutive` | Mean cosine similarity between adjacent query vectors |
| `effective_rank_Q` | Shannon entropy of normalized singular values of Q matrix |
| `attention_entropy` | Shannon entropy of the flattened attention map |
| `query_key_sim_mean` | Mean cosine similarity between query and key vectors |
| `max_attention_weight` | Maximum single attention weight (concentration metric) |
| `attention_variance_per_query` | Mean variance of attention weights per query position |
| `rank_attention_matrix` | Effective rank of the attention matrix via SVD |

---

## Architecture

### Eager Eviction Pattern

Memory is the primary constraint on Apple Silicon M2. We adopt an **Eager Eviction** strategy:

1. **Compute on-the-fly**: Q and K are computed from hidden states and weight projections, not pre-computed and stored.
2. **Process sequentially**: Layers are processed one at a time, never holding multiple layers' tensors simultaneously.
3. **Explicit deletion**: After each layer, all large tensors (H, W_q, W_k, A) are explicitly `del`'ed.
4. **Cache clearing**: `gc.collect()` and `torch.mps.empty_cache()` (for MPS) are called to ensure memory is truly freed.
5. **Per-head computation**: Features are computed immediately after creating each `HeadContext`, then the head tensors (Q, K) are deleted.

### Module Organization

```
core/
├── __init__.py              # Package initialization
├── context.py               # HeadContext dataclass
├── features_library.py      # Mathematical feature functions + FEATURE_REGISTRY
├── analyzer.py              # LightweightAttentionAnalyzer class
└── dataset_manager.py       # DatasetManager for safe persistent storage

main.py                       # Entry point script
README.md                     # This file
```

### Data Flow

```
User Command (main.py)
    ↓
LightweightAttentionAnalyzer.analyze_prompt()
    ├─ Tokenize prompt
    ├─ Forward pass (output_attentions=True, output_hidden_states=True)
    ├─ For each layer:
    │  ├─ Extract hidden state H
    │  ├─ Extract W_q, W_k from model weights
    │  └─ For each head:
    │     ├─ Compute Q, K on-the-fly
    │     ├─ Extract attention map A
    │     ├─ Create HeadContext
    │     ├─ Run all features via get_all_features()
    │     └─ Collect results
    │  └─ Delete tensors, call gc.collect() and empty_cache()
    ↓
DatasetManager.append_records()
    ├─ Load existing dataset (if any)
    ├─ Concatenate new records
    └─ Write to disk (Parquet or CSV)
```

---

## Setup

### Prerequisites

- **Python >= 3.8**
- **macOS with Apple Silicon** (M1/M2/M3) or Linux/Windows with NVIDIA GPU or CPU fallback

### Installation

1. **Clone or create the project directory**:
   ```bash
   cd ~/Desktop/attention-matrices/sparse-attention-patterns
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install transformers>=4.30.0 pandas pyarrow numpy scikit-learn
   ```

   **Note**: The PyTorch command above installs the CPU version. For **optimized Apple Silicon support**, install the MPS-enabled build:
   ```bash
   pip install torch torchvision torchaudio
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```

### Model Download

The first time you run the pipeline, HuggingFace will download the model. Subsequent runs will use the cached version.

To pre-download a model without running analysis:
```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model_name='Qwen/Qwen2.5-0.5B-Instruct'; AutoModelForCausalLM.from_pretrained(model_name); AutoTokenizer.from_pretrained(model_name)"
```

---

## Quick Start

### Minimal Example

```bash
python main.py --prompt "Hello, world!" --output features.parquet
```

This will:
1. Load the default model (Qwen2.5-0.5B-Instruct).
2. Analyze all layers and heads.
3. Save/append results to `features.parquet`.

### Check Dataset Info

```bash
python main.py --info --output features.parquet
```

Output:
```
[INFO] Dataset info:
  Shape: (128, 12)
  Columns: ['model_name', 'layer_idx', 'head_idx', 'prompt_len', 'diagonal_mass_5', ...]
  Null counts: {...}
```

---

## Usage

### Basic Command

```bash
python main.py --prompt "YOUR_PROMPT" [OPTIONS]
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prompt` | str | *(required)* | Input prompt to analyze |
| `--model` | str | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model identifier |
| `--output` | str | `features.parquet` | Output dataset file path |
| `--max-length` | int | 128 | Max token sequence length (truncate if needed) |
| `--layers` | int list | *(all)* | Specific layer indices to analyze (space-separated) |
| `--heads` | int list | *(all)* | Specific head indices to analyze (space-separated) |
| `--device` | str | *(auto)* | `mps`, `cuda`, or `cpu` (auto-detected if not set) |
| `--local-files-only` | flag | False | Only use cached models (no download) |
| `--format` | str | `parquet` | Output format: `parquet` or `csv` |
| `--info` | flag | False | Print dataset info and exit |

### Example Workflows

#### Example 1: Analyze First 3 Layers, All Heads

```bash
python main.py \
  --prompt "The meaning of life is" \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output my_features.parquet \
  --max-length 64 \
  --layers 0 1 2
```

#### Example 2: Analyze Specific Heads in a Specific Layer

```bash
python main.py \
  --prompt "Once upon a time" \
  --layers 5 \
  --heads 0 1 2 3 \
  --output my_features.parquet
```

#### Example 3: Use CSV Format and Inspect Dataset Before Running

```bash
python main.py --info --output my_features.csv --format csv
python main.py \
  --prompt "Attention is all you need" \
  --output my_features.csv \
  --format csv
```

#### Example 4: Use Larger Model (with Local Cache)

```bash
# Pre-download model (only first time)
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')"

# Then run with local-files-only
python main.py \
  --prompt "Your prompt" \
  --model meta-llama/Llama-2-7b-hf \
  --local-files-only \
  --output large_model_features.parquet
```

---

## How to Add a New Feature

### Step-by-Step Guide

Adding a new mathematical feature to the pipeline requires three simple steps:

#### 1. Write the Feature Function

Open [core/features_library.py](core/features_library.py) and add your function in the **Core Feature Functions** section.

**Template**:
```python
def compute_my_feature(ctx: "HeadContext") -> float:
    """
    Brief one-line description of what this feature computes.
    
    Longer explanation (2-3 sentences) describing the mathematical
    interpretation and use cases.
    
    Args:
        ctx: HeadContext instance containing all necessary tensors.
    
    Returns:
        float: The computed scalar value, or np.nan if computation fails.
    
    Mathematical Definition:
        [LaTeX or pseudocode description of the math]
    """
    try:
        # Your computation here
        Q = ctx.Q  # (seq_len, head_dim)
        K = ctx.K  # (seq_len, head_dim)
        A = ctx.attention_map  # (seq_len, seq_len)
        
        # Example: compute something
        result = some_numerical_computation(Q, K, A)
        
        return float(result.item())
    except Exception as e:
        print(f"Error in compute_my_feature: {e}")
        return np.nan
```

**Key Points**:
- Always return a **scalar `float`** (use `.item()` to extract from tensors).
- If computation fails, return **`np.nan`** (not `None`).
- Handle both PyTorch tensors and NumPy arrays gracefully.
- Add an informative docstring with mathematical definition.
- Catch and log exceptions—**never let your feature crash the pipeline**.

**Available Tensors in `ctx`**:
- `ctx.Q`: Query matrix, shape `(seq_len, head_dim)`
- `ctx.K`: Key matrix, shape `(seq_len, head_dim)`
- `ctx.attention_map`: Softmax attention weights, shape `(seq_len, seq_len)`
- `ctx.H_input`: Hidden state input to this layer, shape `(seq_len, hidden_dim)`
- `ctx.W_q`: Query projection matrix, shape `(hidden_dim, hidden_dim)`
- `ctx.W_k`: Key projection matrix, shape `(hidden_dim, hidden_dim)`

#### 2. Register the Feature

In the same file, find the **Feature Registry** section near the bottom and add an entry to `FEATURE_REGISTRY`:

```python
FEATURE_REGISTRY: Dict[str, Callable] = {
    # ... existing features ...
    "my_feature": compute_my_feature,  # <-- ADD THIS LINE
}
```

The **key** (string) is the column name in your output dataset. Choose a descriptive, lowercase, snake_case name.

#### 3. Re-run the Pipeline

Simply run `main.py` with a new prompt or set of parameters:

```bash
python main.py --prompt "New analysis" --output features.parquet
```

Your new feature will automatically:
- Be computed for all heads across all specified layers.
- Appended as a new column to the Parquet/CSV file.
- Appear in `--info` output.

**That's it!** No retraining, no complex setup—just add the function, register it, and run.

### Feature Development Tips

1. **Test locally first**: Use a small prompt and subset of layers:
   ```bash
   python main.py --prompt "test" --output test.parquet --layers 0 --heads 0
   ```

2. **Inspect intermediate results**: Add print statements (they'll appear in console):
   ```python
   print(f"[DEBUG] Q shape: {Q.shape}, K shape: {K.shape}")
   ```

3. **Use caching if expensive**: If your feature computation is expensive, use:
   ```python
   cached = ctx.get_cached_feature("my_feature")
   if cached is not None:
       return cached
   # ... do expensive computation ...
   ctx.set_cached_feature("my_feature", result)
   return result
   ```

4. **Device handling**: Move tensors to CPU for operations not supported on MPS:
   ```python
   Q_cpu = ctx.Q.cpu()
   # ... computation ...
   result_cpu.to(ctx.Q.device)  # move back if needed
   ```

---

## How to Add a New Prompt

Simply pass a new prompt to `main.py`:

```bash
python main.py --prompt "Your new prompt here" --output features.parquet
```

The pipeline will:
1. Tokenize your prompt.
2. Run the forward pass.
3. Extract and compute all features.
4. **Append** the new records to `features.parquet` (preserving all existing rows).

### Handling Long Prompts

If your prompt exceeds the default max length (128 tokens), increase it:

```bash
python main.py \
  --prompt "A very long prompt with many tokens..." \
  --max-length 512 \
  --output features.parquet
```

### Batch Processing Multiple Prompts

To analyze multiple prompts sequentially (appending to the same dataset):

```bash
# Create a simple bash script
cat > analyze_prompts.sh << 'EOF'
#!/bin/bash
prompts=(
  "The quick brown fox"
  "Attention is all you need"
  "Transformers revolutionized NLP"
)
for prompt in "${prompts[@]}"; do
  python main.py --prompt "$prompt" --output features.parquet
done
EOF
chmod +x analyze_prompts.sh
./analyze_prompts.sh
```

---

## How to Add a New Model

The pipeline automatically supports most HuggingFace transformer architectures. Here's how to use a different model:

### Supported Architectures

- **Out-of-the-box**: Qwen, LLaMA, Mistral, Phi, GPT-2, T5, and most standard HuggingFace models that follow the common `model.model.layers[i].self_attn` pattern.

### Using a Different Model

Simply pass the HuggingFace model ID:

```bash
python main.py \
  --prompt "Your prompt" \
  --model meta-llama/Llama-2-7b-hf \
  --output llama_features.parquet
```

Common models:
- `meta-llama/Llama-2-7b-hf` (LLaMA 2, 7B)
- `microsoft/phi-2` (Phi-2, 2.7B)
- `mistralai/Mistral-7B` (Mistral 7B)
- `gpt2` (OpenAI GPT-2)

### Adding Support for Custom/Unsupported Architectures

If a model doesn't work out of the box, the issue is likely in how attention modules are accessed. Edit [core/analyzer.py](core/analyzer.py) and modify the `_get_attention_module()` method:

```python
def _get_attention_module(self, layer_idx: int) -> Optional[Any]:
    """
    Extract the attention module from a specific layer.
    
    Add architecture-specific patterns here.
    """
    try:
        # Standard pattern (LLaMA, Qwen, Mistral, etc.)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx].self_attn
        
        # Add custom pattern for your model here:
        # elif hasattr(self.model, "your_model_attr"):
        #     return self.model.your_model_attr.layers[layer_idx].attention
        
        # Fallback for other architectures
        elif hasattr(self.model, "transformer"):
            return self.model.transformer.h[layer_idx].attn
        else:
            return None
    except (AttributeError, IndexError):
        return None
```

**To find the correct path**:
```python
# In Python interpreter
from transformers import AutoModel
model = AutoModel.from_pretrained("your_model_id")
print(model)  # Inspect the structure to find attention module paths
```

---

## Module Reference

### `core/context.py` — `HeadContext`

Dataclass encapsulating the state of a single attention head.

**Key Methods**:
- `get_head_dim() -> int`: Return the feature dimension of this head.
- `clear_cache() -> None`: Clear the feature cache.
- `get_cached_feature(feature_name: str) -> Optional[Any]`: Retrieve cached value.
- `set_cached_feature(feature_name: str, value: Any) -> None`: Store computed value.

### `core/analyzer.py` — `LightweightAttentionAnalyzer`

Main class for attention analysis with memory-efficient eager eviction.

**Key Methods**:
- `__init__(model_name, device=None, local_files_only=False)`: Initialize and load model.
- `analyze_prompt(prompt, max_length=128, layer_indices=None, head_indices=None) -> List[Dict]`: Run full analysis pipeline.

### `core/dataset_manager.py` — `DatasetManager`

Handles safe persistent storage of feature records.

**Key Methods**:
- `read_dataset() -> Optional[pd.DataFrame]`: Load existing dataset.
- `append_records(records: List[Dict]) -> int`: Append and persist new records.
- `get_dataset_info() -> Optional[Dict]`: Get summary statistics.
- `delete_dataset() -> bool`: Delete the dataset file.

### `core/features_library.py`

Pure mathematical feature functions.

**Key Functions**:
- `get_all_features(ctx) -> Dict[str, float]`: Compute all registered features for a head.
- `FEATURE_REGISTRY`: Dictionary of all available features.

---

## Best Practices

1. **Memory Management**:
   - Always use `--layers` and `--heads` to limit analysis scope if memory is tight.
   - Start with small prompts (truncate to ~64 tokens) when developing.

2. **Reproducibility**:
   - Store your command-line invocations in shell scripts:
     ```bash
     #!/bin/bash
     python main.py --prompt "Analysis A" --output run_a.parquet
     python main.py --prompt "Analysis B" --output run_b.parquet
     ```

3. **Data Integrity**:
   - Always back up your Parquet/CSV files before large batch runs.
   - Use `--info` to verify dataset state before appending.

4. **Feature Development**:
   - Write features as pure functions (no side effects).
   - Test with a minimal dataset first: `--layers 0 --heads 0 --max-length 32`.
   - Use numerical stability tricks (add epsilon to prevent log(0), etc.).

5. **Performance**:
   - CPU is typically faster than MPS for very small models (<1B params).
   - For GPU inference, use `--device cuda`.
   - Parquet format is faster and more compact than CSV for large datasets.

---

## Troubleshooting

### "No module named 'transformers'"
**Solution**: Reinstall dependencies: `pip install transformers transformers pandas pyarrow`

### "CUDA out of memory" / "MPS out of memory"
**Solution**:
- Reduce max-length: `--max-length 64`
- Analyze fewer layers: `--layers 0 1 2`
- Use a smaller model: `--model Qwen/Qwen2.5-0.5B-Instruct`
- Switch to CPU: `--device cpu`

### "Model not found" or "Timeout downloading"
**Solution**:
- Pre-download with: `python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('model_id')"`
- Use local cache: `--local-files-only`
- Check your internet connection.

### "AttentionModule is None"
**Solution**: The architecture isn't recognized. Check [core/analyzer.py](core/analyzer.py) `_get_attention_module()` and add support for your model's structure.

### Feature returns `nan` for all heads
**Solution**:
- Check console for error messages
- Verify attention_map shape: `print(A.shape)` in your feature function
- Test with a simple feature like `max_attention_weight` to isolate the issue

### Parquet/CSV file grows but features don't change
**Solution**: You likely added a new feature but the existing dataset doesn't have that column. The new feature will only appear for rows analyzed *after* registration. Old rows will have `NaN` in the new column (auto-filled during concat).

---

## Citation & Acknowledgments

This pipeline is designed for research on sparse attention mechanisms and RoPE (Rotary Position Embedding) in transformer architectures.

**Recommended citation format** (if you use this):
```bibtex
@software{sparse_attention_analyzer_2026,
  title={Sparse Attention Analysis Pipeline},
  author={[Your Name]},
  year={2026},
  url={https://github.com/[your_repo]}
}
```

---

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

---

For questions, issues, or contributions, please open an issue or contact the maintainers.

**Happy analyzing!** 🚀
