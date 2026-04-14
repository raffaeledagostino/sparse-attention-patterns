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
import inspect
import warnings
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import ATTN_IMPLEMENTATION


_orig_is_autocast_enabled = torch.is_autocast_enabled


def _compat_is_autocast_enabled(device_type=None):
    """Compatibility wrapper for torch/transformers autocast signature mismatches."""
    try:
        if device_type is None:
            return _orig_is_autocast_enabled()
        return _orig_is_autocast_enabled(device_type)
    except TypeError:
        return _orig_is_autocast_enabled()
    except RuntimeError:
        return _orig_is_autocast_enabled()


if torch.is_autocast_enabled is not _compat_is_autocast_enabled:
    torch.is_autocast_enabled = _compat_is_autocast_enabled

from core.context import HeadContext
from core.features_library import get_all_features, FEATURE_REGISTRY


class LightweightAttentionAnalyzer:
    """
    Analyzes attention mechanisms in causal language models with memory efficiency.
    
    This class implements the complete pipeline for extracting mathematical features
    from attention heads in transformer models, with special emphasis on memory
    efficiency for Apple Silicon through eager tensor eviction and explicit
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
        _device_map = {
            "mps":  None,   
            "cuda": "auto", 
            "cpu":  "cpu",
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype="auto",
                device_map=_device_map[self.device],
                trust_remote_code=True,
                attn_implementation=ATTN_IMPLEMENTATION,
                local_files_only=local_files_only,
            )
            if self.device == "mps":
                self.model = self.model.to("mps")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")
        
        # Ensure pad token is set for proper tokenization
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "attn_implementation"):
                self.model.config.attn_implementation = ATTN_IMPLEMENTATION
            if hasattr(self.model.config, "_attn_implementation"):
                self.model.config._attn_implementation = ATTN_IMPLEMENTATION
        print(f"[Analyzer] Model loaded successfully. Using {self.model.config.num_hidden_layers} layers.")
    
    def _is_qwen2(self) -> bool:
        return getattr(self.model.config, "model_type", "").lower() == "qwen2"


    def _forward_with_qwen2_rope_capture(self, input_ids):
        """Same logic as Qwen3, but patches modeling_qwen2."""
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

        q_captures, k_captures = {}, {}
        call_counter = [0]
        orig_rope = qwen2_mod.apply_rotary_pos_emb

        def _capturing_rope(q, k, cos, sin, *args, **kwargs):
            rotated_q, rotated_k = orig_rope(q, k, cos, sin, *args, **kwargs)
            layer_idx = call_counter[0]
            q_captures[layer_idx] = rotated_q.detach().squeeze(0).permute(1, 0, 2).cpu()
            k_captures[layer_idx] = rotated_k.detach().squeeze(0).permute(1, 0, 2).cpu()
            call_counter[0] += 1
            return rotated_q, rotated_k

        qwen2_mod.apply_rotary_pos_emb = _capturing_rope
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            qwen2_mod.apply_rotary_pos_emb = orig_rope

        return outputs, q_captures, k_captures




    def _is_qwen3(self) -> bool:
        """Detect Qwen3 architecture via model config."""
        model_type = getattr(self.model.config, "model_type", "").lower()
        return "qwen3" in model_type


    def _forward_with_qwen3_rope_capture(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[CausalLMOutputWithPast, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Forward pass for Qwen3 that intercepts Q and K post-QK-Norm and post-RoPE.

        Monkey-patches `apply_rotary_pos_emb` in the qwen3 modeling module for
        the duration of the forward pass, capturing the rotated tensors per layer.
        The patch is ALWAYS restored in the finally block, even on exception.

        Returns:
            outputs:     Standard HuggingFace CausalLMOutputWithPast.
            q_captures:  Dict[layer_idx -> Tensor(seq_len, num_q_heads, head_dim)]
            k_captures:  Dict[layer_idx -> Tensor(seq_len, num_kv_heads, head_dim)]

        Notes:
            - apply_rotary_pos_emb is called exactly once per layer in Qwen3,
            so call_counter maps 1-to-1 to layer index.
            - Q and K are detached immediately to avoid retaining the graph.
            - RoPE is NOT applied to the fallback pre-rope path (other models).
        """
        import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

        q_captures: Dict[int, torch.Tensor] = {}
        k_captures: Dict[int, torch.Tensor] = {}
        call_counter = [0]  # list for mutability inside closure

        orig_rope = qwen3_mod.apply_rotary_pos_emb

        def _capturing_rope(q, k, cos, sin, *args, **kwargs):
            rotated_q, rotated_k = orig_rope(q, k, cos, sin, *args, **kwargs)
            layer_idx = call_counter[0]
            # shape from model: (batch=1, heads, seq_len, head_dim)
            # → target shape: (seq_len, heads, head_dim)
            q_captures[layer_idx] = rotated_q.detach().squeeze(0).permute(1, 0, 2).cpu()
            k_captures[layer_idx] = rotated_k.detach().squeeze(0).permute(1, 0, 2).cpu()
            call_counter[0] += 1
            return rotated_q, rotated_k

        qwen3_mod.apply_rotary_pos_emb = _capturing_rope
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            qwen3_mod.apply_rotary_pos_emb = orig_rope  # always restore

        return outputs, q_captures, k_captures

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
    
    def analyze_prompt(self, prompt, max_length=128, layer_indices=None,
                    head_indices=None, prompt_source="unknown"):
        
        tokens = self.tokenizer(prompt, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length)
        input_ids = tokens["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        print(f"[Analyzer] Tokenized to {seq_len} tokens.")

        # -------------------------------------------------------------------------
        # Forward pass — branch on architecture
        # -------------------------------------------------------------------------
        print(f"[Analyzer] Running forward pass...")
        try:
            if self._is_qwen3():
                outputs, q_post_rope, k_post_rope = self._forward_with_qwen3_rope_capture(input_ids)
                q_k_pre_rope = False
            elif self._is_qwen2():
                outputs, q_post_rope, k_post_rope = self._forward_with_qwen2_rope_capture(input_ids)
                q_k_pre_rope = False
            else:
                # Llama3, Mistral, ecc. — fallback
                with torch.no_grad():
                    outputs = self.model(input_ids, output_attentions=True,
                                        output_hidden_states=True, return_dict=True)
                q_post_rope, k_post_rope = {}, {}
                q_k_pre_rope = True
                print(f"[Analyzer] Non-Qwen3 model: Q/K will be pre-RoPE.")
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {e}")

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        if not attentions:
            raise RuntimeError(
                "Forward pass did not return attention tensors. "
                "Ensure the model runs with eager attention implementation."
            )

        num_layers = len(attentions)
        num_heads  = attentions[0].shape[1]
        if layer_indices is None:
            layer_indices = list(range(num_layers))
        else:
            layer_indices = [i for i in layer_indices if 0 <= i < num_layers]
        if head_indices is None:
            head_indices = list(range(num_heads))
        else:
            head_indices = [i for i in head_indices if 0 <= i < num_heads]

        results = []

        try:
            for layer_idx in layer_indices:
                print(f"[Analyzer]   Layer {layer_idx}/{num_layers - 1}...")

                H_input = hidden_states[layer_idx].squeeze(0)
                attention_module = self._get_attention_module(layer_idx)
                if attention_module is None:
                    print(f"[Analyzer]   Skipping layer {layer_idx}: module not found.")
                    continue

                W_q, W_k, W_v = self._extract_weight_projections(attention_module)
                if W_q is None:
                    print(f"[Analyzer]   Skipping layer {layer_idx}: projections not found.")
                    continue

                num_q_heads = getattr(attention_module, "num_heads",
                            getattr(attention_module, "num_attention_heads",
                                    self.model.config.num_attention_heads))
                num_kv_heads = getattr(attention_module, "num_key_value_heads",
                            getattr(self.model.config, "num_key_value_heads", num_q_heads))
                head_dim = getattr(attention_module, "head_dim",
                        getattr(self.model.config, "head_dim",
                                W_q.shape[0] // num_q_heads))
                kv_group_size = max(1, num_q_heads // max(1, num_kv_heads))

                W_q_heads = W_q.reshape(num_q_heads, head_dim, -1)
                W_k_heads = W_k.reshape(num_kv_heads, head_dim, -1)
                W_v_heads = W_v.reshape(num_kv_heads, head_dim, -1)

                # ------------------------------------------------------------------
                # Q/K source: post-RoPE if captured (Qwen3), else manual projection
                # with QK-Norm applied where available.
                # ------------------------------------------------------------------
                if layer_idx in q_post_rope:
                    # Qwen3: post-QK-Norm, post-RoPE — consistent with attention_map
                    Q_all = q_post_rope[layer_idx].to(H_input.device)  # (seq, q_heads, head_dim)
                    K_all = k_post_rope[layer_idx].to(H_input.device)  # (seq, kv_heads, head_dim)
                else:
                    # Fallback: manual projection + QK-Norm (no RoPE)
                    Q_raw = H_input @ W_q.T
                    if hasattr(attention_module, "q_proj") and attention_module.q_proj.bias is not None:
                        Q_raw = Q_raw + attention_module.q_proj.bias
                    Q_all = Q_raw.reshape(seq_len, num_q_heads, head_dim)

                    K_raw = H_input @ W_k.T
                    if hasattr(attention_module, "k_proj") and attention_module.k_proj.bias is not None:
                        K_raw = K_raw + attention_module.k_proj.bias
                    K_all = K_raw.reshape(seq_len, num_kv_heads, head_dim)

                    # Apply QK-Norm if present (e.g. non-Qwen3 model with QK-Norm)
                    if hasattr(attention_module, "q_norm") and attention_module.q_norm is not None:
                        with torch.no_grad():
                            Q_all = attention_module.q_norm(Q_all)
                    if hasattr(attention_module, "k_norm") and attention_module.k_norm is not None:
                        with torch.no_grad():
                            K_all = attention_module.k_norm(K_all)

                    del Q_raw, K_raw  # eager eviction

                qk_norm_gamma = None
                if hasattr(attention_module, "q_norm") and \
                hasattr(attention_module.q_norm, "weight"):
                    qk_norm_gamma = attention_module.q_norm.weight.detach()

                layer_attentions = attentions[layer_idx].squeeze(0)

                for head_idx in head_indices:
                    kv_head_idx = head_idx // kv_group_size if num_kv_heads < num_q_heads \
                                else head_idx
                    kv_head_idx = min(kv_head_idx, num_kv_heads - 1)

                    Q = Q_all[:, head_idx, :]
                    K = K_all[:, kv_head_idx, :]
                    attention_map = layer_attentions[head_idx, :, :]

                    ctx = HeadContext(
                        model_name=self.model_name,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        prompt_len=seq_len,
                        H_input=H_input,
                        W_q=W_q_heads[head_idx],
                        W_k=W_k_heads[kv_head_idx],
                        W_v=W_v_heads[kv_head_idx],
                        Q=Q,
                        K=K,
                        attention_map=attention_map,
                        rmsnorm_gamma=qk_norm_gamma,
                        q_k_pre_rope=q_k_pre_rope,  # flag di documentazione
                    )

                    features = get_all_features(ctx)
                    result = {
                        "model_name":    self.model_name,
                        "layer_idx":     layer_idx,
                        "head_idx":      head_idx,
                        "prompt_len":    seq_len,
                        "prompt_source": prompt_source,
                        "q_k_pre_rope":  q_k_pre_rope,
                    }
                    result.update(features)
                    results.append(result)

                    del Q, K, ctx

                del H_input, W_q, W_k, W_v, W_q_heads, W_k_heads, W_v_heads, \
                    Q_all, K_all, layer_attentions, attention_module
                gc.collect()
                if self.device == "mps":   torch.mps.empty_cache()
                elif self.device == "cuda": torch.cuda.empty_cache()

        finally:
            del hidden_states, attentions, outputs, input_ids, tokens
            # Libera anche i capture (potrebbero essere grandi su seq_len lunghe)
            q_post_rope.clear()
            k_post_rope.clear()
            gc.collect()
            if self.device == "mps":   torch.mps.empty_cache()
            elif self.device == "cuda": torch.cuda.empty_cache()

        print(f"[Analyzer] Completed. Extracted {len(results)} head-level feature sets.")
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
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract query, key, and value weight projections from an attention module.
        
        Args:
            attention_module: The attention module (self_attn).
        
        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
                (W_q, W_k, W_v) weight matrices, or (None, None, None) if extraction fails.
        
        Notes:
            Returned tensors are detached and remain on the model device.
        """
        try:
            # Extract weight matrices
            if (
                hasattr(attention_module, "q_proj")
                and hasattr(attention_module, "k_proj")
                and hasattr(attention_module, "v_proj")
            ):
                W_q = attention_module.q_proj.weight  # shape: (hidden_dim, hidden_dim)
                W_k = attention_module.k_proj.weight  # shape: (hidden_dim, hidden_dim)
                W_v = attention_module.v_proj.weight
            elif (
                hasattr(attention_module, "q_proj_weight")
                and hasattr(attention_module, "k_proj_weight")
                and hasattr(attention_module, "v_proj_weight")
            ):
                W_q = attention_module.q_proj_weight
                W_k = attention_module.k_proj_weight
                W_v = attention_module.v_proj_weight
            else:
                return None, None, None
            
            # Ensure tensors are on the correct device and detached
            W_q = W_q.detach()
            W_k = W_k.detach()
            W_v = W_v.detach()
            
            return W_q, W_k, W_v
        except (AttributeError, RuntimeError):
            return None, None, None
