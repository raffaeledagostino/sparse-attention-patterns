# Sparse Attention — Dataset & Usage

This repository extracts per-head attention features from transformer models and writes them to a modular dataset (Parquet/CSV). It includes tools to run a single prompt analysis, add new features, and perform basic EDA and a target-feature prediction experiment.

Structure 
- `core/` — core implementation: `analyzer.py` (pipeline), `context.py` (HeadContext), `features_library.py` (feature functions).
- `data/persistence.py` — safe append + deduplication to Parquet/CSV.
- `pipeline.py` / `main.py` — thin CLI/runner for single-prompt runs.
- `eda.py` and the notebooks (`*.ipynb`) — exploratory analysis and the target prediction task.

What the code does
- Tokenize a prompt and run a forward pass with `output_attentions=True` / `output_hidden_states=True`.
- For each layer: compute Q/K for all heads, slice per-head, compute feature functions, and emit one record per (model, prompt, layer, head).
- Each run appends its rows to the dataset file; the persistence layer deduplicates on the chosen primary key so re-runs are safe.

Dataset format
- Each row contains metadata (`model_name`, `prompt_id` or text, `layer_idx`, `head_idx`, `prompt_len`) plus one column per feature.
- The dataset is modular: adding a new feature simply adds a new column for future runs; historical rows will show NaN for that column until re-run.

Adding features 
1. Implement a pure function in `core/features_library.py` with signature `def my_feature(ctx: HeadContext) -> float`.
2. Return a scalar `float` or `np.nan` on failure; use `ctx.cache` for expensive intermediate results.
3. Register the function in `FEATURE_REGISTRY` (the dictionary mapping column names to callables).
No training or pipeline changes are required — new features compute on the fly during analysis.

Models and device support
- The code was developed with models similar in structure to Qwen3 and Mistral v0.7 (both tested here). It will work for other HuggingFace models that expose per-layer attention modules (common pattern: `model.model.layers[i].self_attn`).
- Device selection is automatic (MPS > CUDA > CPU) but you can force `--device` on the CLI. Right now it is built to be used on CUDA, specifically in Kaggle.

EDA and target prediction
- The `eda.py` script and notebooks show feature distributions, correlations, and basic visualizations.
- A target-feature prediction task (notebook `lightGBM_opt&fitting.ipynb`) demonstrates training a simple regressor/classifier on the extracted features to predict a chosen target metric.


Notes & best practices
- Keep `--max-length` modest on memory-constrained devices — attention is O(seq_len^2).
- Use `--local-files-only` if you want to run with cached models only.
- To backfill a new feature across old prompts, re-run the analyzer on those prompts or run a batch script to reprocess them.


----

