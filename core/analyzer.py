"""
Lightweight attention analyzer for extracting features from LLM attention heads.

This module implements the LightweightAttentionAnalyzer class, which handles
the complete pipeline of loading an LLM, executing a forward pass with
attention extraction, and computing mathematical features on a per-head basis.

The key architectural feature is "Eager Eviction": tensors are explicitly
deleted after use, and garbage collection and torch.mps.empty_cache() are
called after each layer to prevent out-of-memory errors on Apple Silicon.
"""

from typing import List, Dict, Tuple, Optional, Any
import gc
import warnings
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from core.context import HeadContext
from core.features_library import get_all_features, FEATURE_REGISTRY


class LightweightAttentionAnalyzer:
    """
    Analyzes attention mechanisms in causal language models with memory efficiency.
    
    This class implements the complete pipeline for extracting mathematical features
    from attention heads in transformer models, with special emphasis on memory
    efficiency for Apple Silicon (M2) through eager tensor eviction and explicit
    garbage collection.
    
    Key Features:
      - Automatic prompt tokenization and batching.
      - Layer-wise hidden state extraction.
      - On-the-fly Q and K computation from hidden states and weight projections.
      - Grouped Query Attention (GQA) support via head replication.
      - All registered features computed per head per layer.
      - Strict memory management: explicit tensor deletion and cache clearing.
    
    Attributes:
        model_name (str): HuggingFace model identifier.
        model: The loaded PyTorch model.
        tokenizer: Associated tokenizer.
        device (str): Computation device ("mps" for Apple Silicon, "cpu", or "cuda").
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ):
        """
        Initialize the analyzer and load the model and tokenizer.
        
        Args:
            model_name (str): HuggingFace model identifier (e.g., "Qwen/Qwen2.5-0.5B-Instruct").
            device (Optional[str]): Device to use ("mps", "cpu", "cuda"). Auto-detected if None.
            local_files_only (bool): If True, only use locally cached models.
        
        Raises:
            RuntimeError: If model/tokenizer cannot be loaded.
        """
        self.model_name = model_name
        self.device = device or self._detect_device()
        
        print(f"[Analyzer] Loading model '{model_name}' on device '{self.device}'...")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto" if self.device != "cpu" else "cpu",
                local_files_only=local_files_only,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")
        
        # Ensure pad token is set for proper tokenization
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print(f"[Analyzer] Model loaded successfully. Using {self.model.config.num_hidden_layers} layers.")
    
    @staticmethod
    def _detect_device() -> str:
        """
        Detect the appropriate device for computation.
        
        Returns:
            str: "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" fallback.
        """
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def analyze_prompt(
        self,
        prompt: str,
        max_length: int = 128,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze a prompt and extract features from specified attention heads.
        
        This is the main entry point. It tokenizes the prompt, runs the forward pass,
        extracts hidden states and weights, computes Q and K on-the-fly, and
        computes all registered features for each attention head.
        
        Args:
            prompt (str): The input text to analyze.
            max_length (int): Maximum tokens to process (truncate if needed).
            layer_indices (Optional[List[int]]): Which layers to analyze. If None, analyze all.
            head_indices (Optional[List[int]]): Which heads within a layer to analyze. If None, analyze all.
        
        Returns:
            List[Dict[str, Any]]: List of feature dictionaries, one per (layer, head) pair.
                                  Each dict contains metadata and computed features.
        
        Raises:
            RuntimeError: If forward pass fails.
        """
        print(f"\n[Analyzer] Processing prompt: '{prompt[:80]}...'")
        
        # Tokenize prompt
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = tokens["input_ids"].to(self.device)
        seq_len = input_ids.shape[1]
        
        print(f"[Analyzer] Tokenized to {seq_len} tokens.")
        
        # Forward pass with attention extraction
        print(f"[Analyzer] Running forward pass...")
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {e}")
        
        # Extract hidden states and attentions
        hidden_states = outputs.hidden_states  # Tuple of length num_layers + 1 (includes embedding)
        attentions = outputs.attentions  # Tuple of length num_layers
        
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        
        if layer_indices is None:
            layer_indices = list(range(num_layers))
        else:
            layer_indices = [i for i in layer_indices if 0 <= i < num_layers]
        
        if head_indices is None:
            head_indices = list(range(num_heads))
        else:
            head_indices = [i for i in head_indices if 0 <= i < num_heads]
        
        print(f"[Analyzer] Analyzing {len(layer_indices)} layers x {len(head_indices)} heads = {len(layer_indices) * len(head_indices)} total heads.")
        
        # Extract weight projections (once, not per layer)
        results = []
        
        try:
            for layer_idx in layer_indices:
                print(f"[Analyzer]   Layer {layer_idx}/{num_layers - 1}...")
                
                # Get hidden state for this layer (input to this layer's attention)
                H_input = hidden_states[layer_idx]  # shape: (batch_size, seq_len, hidden_dim)
                H_input = H_input.squeeze(0)  # Remove batch dimension
                
                # Get layer module to extract weights
                # For standard transformer architectures, attention is in self_attn
                attention_module = self._get_attention_module(layer_idx)
                if attention_module is None:
                    print(f"[Analyzer]   Could not access attention module for layer {layer_idx}. Skipping.")
                    continue
                
                # Extract or compute W_q and W_k
                W_q, W_k = self._extract_weight_projections(attention_module)
                if W_q is None or W_k is None:
                    print(f"[Analyzer]   Could not extract weight projections for layer {layer_idx}. Skipping.")
                    continue
                
                hidden_dim = W_q.shape[0]
                num_q_heads = attention_module.num_heads
                num_kv_heads = getattr(attention_module, "num_key_value_heads", num_q_heads)
                head_dim = hidden_dim // num_q_heads
                
                # Get attention weights for this layer
                layer_attentions = attentions[layer_idx]  # shape: (batch_size, num_heads, seq_len, seq_len)
                layer_attentions = layer_attentions.squeeze(0)  # Remove batch (assuming batch_size=1)
                
                for head_idx in head_indices:
                    # Compute Q and K on-the-fly
                    Q = H_input @ W_q.T  # (seq_len, hidden_dim)
                    Q = Q.reshape(seq_len, num_q_heads, head_dim)
                    Q = Q[:, head_idx, :]  # Extract this head: (seq_len, head_dim)
                    
                    K = H_input @ W_k.T  # (seq_len, hidden_dim)
                    K = K.reshape(seq_len, num_kv_heads, head_dim)
                    
                    # Handle Grouped Query Attention (GQA): replicate KV heads if needed
                    if num_kv_heads < num_q_heads:
                        kv_head_idx = head_idx % num_kv_heads
                    else:
                        kv_head_idx = head_idx
                    
                    K = K[:, kv_head_idx, :]  # Extract corresponding KV head: (seq_len, head_dim)
                    
                    # Get attention map for this head
                    attention_map = layer_attentions[head_idx, :, :]  # (seq_len, seq_len)
                    
                    # Create HeadContext
                    ctx = HeadContext(
                        model_name=self.model_name,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        prompt_len=seq_len,
                        H_input=H_input,
                        W_q=W_q,
                        W_k=W_k,
                        Q=Q,
                        K=K,
                        attention_map=attention_map,
                    )
                    
                    # Compute all features
                    features = get_all_features(ctx)
                    
                    # Build result dictionary
                    result = {
                        "model_name": self.model_name,
                        "layer_idx": layer_idx,
                        "head_idx": head_idx,
                        "prompt_len": seq_len,
                    }
                    result.update(features)
                    results.append(result)
                    
                    # Explicit cleanup after each head
                    del Q, K, ctx
                
                # Explicit layer cleanup: delete large tensors and call garbage collection
                del H_input, W_q, W_k, layer_attentions, attention_module
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
        
        finally:
            # Final cleanup
            del hidden_states, attentions, outputs, input_ids, tokens
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
        
        print(f"[Analyzer] Completed analysis. Extracted {len(results)} head-level feature sets.")
        return results
    
    def _get_attention_module(self, layer_idx: int) -> Optional[Any]:
        """
        Extract the attention module from a specific layer.
        
        Args:
            layer_idx (int): Index of the transformer layer.
        
        Returns:
            Optional[Any]: The attention module, or None if not found.
        
        Notes:
            This implementation supports standard HuggingFace architectures
            (LLaMA, Qwen, Mistral, etc.) where attention is in model.model.layers[i].self_attn.
        """
        try:
            # Common pattern for HuggingFace transformers
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                return self.model.model.layers[layer_idx].self_attn
            # Fallback for other architectures
            elif hasattr(self.model, "transformer"):
                return self.model.transformer.h[layer_idx].attn
            else:
                return None
        except (AttributeError, IndexError):
            return None
    
    def _extract_weight_projections(
        self, attention_module: Any
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract query and key weight projections from an attention module.
        
        Args:
            attention_module: The attention module (self_attn).
        
        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: (W_q, W_k) weight matrices,
                                                                    or (None, None) if extraction fails.
        
        Notes:
            W_q and W_k have shape (hidden_dim, hidden_dim) and are already on the correct device.
        """
        try:
            # Extract weight matrices
            if hasattr(attention_module, "q_proj") and hasattr(attention_module, "k_proj"):
                W_q = attention_module.q_proj.weight  # shape: (hidden_dim, hidden_dim)
                W_k = attention_module.k_proj.weight  # shape: (hidden_dim, hidden_dim)
            elif hasattr(attention_module, "q_proj_weight") and hasattr(attention_module, "k_proj_weight"):
                W_q = attention_module.q_proj_weight
                W_k = attention_module.k_proj_weight
            else:
                return None, None
            
            # Ensure tensors are on the correct device and detached
            W_q = W_q.detach()
            W_k = W_k.detach()
            
            return W_q, W_k
        except (AttributeError, RuntimeError):
            return None, None
