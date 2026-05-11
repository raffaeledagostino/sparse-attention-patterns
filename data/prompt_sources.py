"""Prompt source abstractions for analysis pipelines."""

import abc
from typing import Optional

import numpy as np
from datasets import load_dataset


class PromptSource(abc.ABC):
    """Common interface for prompt providers."""

    @abc.abstractmethod
    def get_prompt(self, idx: int = 0) -> tuple[str, str]:
        """Return ``(prompt_text, prompt_id)`` for the given index."""

    @property
    @abc.abstractmethod
    def source_tag(self) -> str:
        """Short source identifier used in metadata and prompt_id."""


class DatasetPromptSource(PromptSource):
    """Prompt source backed by a Hugging Face dataset text column."""

    def __init__(
        self,
        tokenizer,
        dataset_name: str,
        target_tokens: int,
        dataset_config: Optional[str] = None,
        split: str = "train",
        text_column: str = "text",
        min_chars: int = 100,
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_column = text_column
        self.target_tokens = target_tokens
        self.min_chars = min_chars
        self._docs = None

    def _load(self) -> None:
        if self._docs is None:
            ds = load_dataset(self.dataset_name, self.dataset_config, split=self.split)
            # Pre-filtra per lunghezza chars prima ancora di tokenizzare
            min_chars_threshold = max(self.min_chars, self.target_tokens * 3)
            self._docs = [
                text for text in ds[self.text_column]
                if isinstance(text, str) and len(text.strip()) >= min_chars_threshold
            ]

    @property
    def source_tag(self) -> str:
        if self.dataset_config is None:
            return f"{self.dataset_name}_{self.split}"
        return f"{self.dataset_name}_{self.dataset_config}_{self.split}"

    def get_prompt(self, idx: int = 0) -> tuple[str, str]:
        self._load()
        count = 0
        for offset, text in enumerate(self._docs):
            ids = self.tokenizer(text, return_tensors="pt").input_ids
            if ids.shape[1] >= self.target_tokens:
                if count == idx:
                    prompt = self.tokenizer.decode(
                        ids[0, :self.target_tokens],
                        skip_special_tokens=True,
                    )
                    prompt_id = f"{self.source_tag}_doc{offset}_{self.target_tokens}tok"
                    return prompt, prompt_id
                count += 1

        raise ValueError(
            f"Not enough documents with >= {self.target_tokens} tokens in '{self.source_tag}'."
        )



class RandomTokenPromptSource(PromptSource):
    """Prompt source based on random token sampling from model vocab."""

    def __init__(
        self,
        tokenizer,
        target_tokens: int,
        seed_base: int = 42,
        exclude_special: bool = True,
    ):
        self.tokenizer = tokenizer
        self.target_tokens = target_tokens
        self.seed_base = seed_base
        self.exclude_special = exclude_special
        self._vocab_ids = None

    def _build_vocab(self) -> None:
        if self._vocab_ids is None:
            all_ids = list(range(self.tokenizer.vocab_size))
            if self.exclude_special:
                special = set(self.tokenizer.all_special_ids)
                all_ids = [tok_id for tok_id in all_ids if tok_id not in special]
            self._vocab_ids = np.array(all_ids, dtype=np.int64)

    @property
    def source_tag(self) -> str:
        return "random_vocab"

    def get_prompt(self, idx: int = 0) -> tuple[str, str]:
        self._build_vocab()
        rng = np.random.default_rng(self.seed_base + idx)
        token_ids = rng.choice(self._vocab_ids, size=self.target_tokens, replace=True)
        prompt = self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        rechecked = self.tokenizer(prompt, return_tensors="pt").input_ids
        if rechecked.shape[1] < self.target_tokens:
            extra = rng.choice(self._vocab_ids, size=self.target_tokens // 2, replace=True)
            token_ids = np.concatenate([token_ids, extra])
            prompt = self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        
        prompt_id = f"random_vocab_seed{self.seed_base + idx}_{self.target_tokens}tok"
        return prompt, prompt_id



import abc
from typing import Optional
import numpy as np
from datasets import load_dataset

class PromptSource(abc.ABC):
    """Common interface for prompt providers."""

    @abc.abstractmethod
    def get_prompt(self, idx: int = 0) -> tuple[str, str]:
        """Return ``(prompt_text, prompt_id)`` for the given index."""

    @property
    @abc.abstractmethod
    def source_tag(self) -> str:
        """Short source identifier used in metadata and prompt_id."""


class StreamingDatasetPromptSource(PromptSource):
    """
    Prompt source backed by a Hugging Face dataset in streaming mode.
    Ideal for massive datasets like FineWeb-Edu where downloading
    the full dataset is not feasible.
    """

    def __init__(
        self,
        tokenizer,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        target_tokens: int = 512,
        dataset_config: Optional[str] = "sample-10BT", # Configurazione raccomandata per test
        split: str = "train",
        text_column: str = "text",
        min_chars: int = 100,
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_column = text_column
        self.target_tokens = target_tokens
        self.min_chars = min_chars
        
        # Cache interna per non ricominciare lo stream ad ogni get_prompt
        self._cached_prompts = []
        self._iterator = None

    @property
    def source_tag(self) -> str:
        tag_name = self.dataset_name.split("/")[-1]
        if self.dataset_config is None:
            return f"{tag_name}_{self.split}_stream"
        return f"{tag_name}_{self.dataset_config}_{self.split}_stream"

    def _initialize_stream(self) -> None:
        """Initialize the Hugging Face streaming dataset iterator."""
        if self._iterator is None:
            ds = load_dataset(
                self.dataset_name, 
                name=self.dataset_config, # FineWeb usa l'argomento 'name' per i dump
                split=self.split, 
                streaming=True
            )
            self._iterator = iter(ds)

    def _fetch_up_to(self, target_idx: int) -> None:
        """Advance the stream until we have at least target_idx + 1 valid prompts."""
        self._initialize_stream()
        
        # Un'euristica rapida per evitare di tokenizzare stringhe palesemente troppo corte
        # In media 1 token = ~4 caratteri. Buffer di 3x per sicurezza
        min_chars_threshold = max(self.min_chars, self.target_tokens * 3)

        while len(self._cached_prompts) <= target_idx:
            try:
                row = next(self._iterator)
            except StopIteration:
                raise ValueError(
                    f"Stream exhausted. Found only {len(self._cached_prompts)} "
                    f"documents with >= {self.target_tokens} tokens in '{self.source_tag}'."
                )

            text = row.get(self.text_column, "")
            
            # Filtro euristico veloce
            if not isinstance(text, str) or len(text.strip()) < min_chars_threshold:
                continue

            # Tokenizzazione vera e propria
            ids = self.tokenizer(text, return_tensors="pt").input_ids
            
            if ids.shape[1] >= self.target_tokens:
                # Decodifica esatta dei primi target_tokens
                prompt = self.tokenizer.decode(
                    ids[0, :self.target_tokens],
                    skip_special_tokens=True,
                )
                
                offset = len(self._cached_prompts)
                prompt_id = f"{self.source_tag}_doc{offset}_{self.target_tokens}tok"
                
                self._cached_prompts.append((prompt, prompt_id))

    def get_prompt(self, idx: int = 0) -> tuple[str, str]:
        """Fetch prompt from cache or advance stream to find it."""
        self._fetch_up_to(idx)
        return self._cached_prompts[idx]