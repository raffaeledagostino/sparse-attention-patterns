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
            self._docs = [
                text
                for text in ds[self.text_column]
                if isinstance(text, str) and len(text.strip()) >= self.min_chars
            ]

    @property
    def source_tag(self) -> str:
        if self.dataset_config is None:
            return f"{self.dataset_name}_{self.split}"
        return f"{self.dataset_name}_{self.dataset_config}_{self.split}"

    def get_prompt(self, idx: int = 0) -> tuple[str, str]:
        self._load()
        for offset, text in enumerate(self._docs[idx:], start=idx):
            ids = self.tokenizer(text, return_tensors="pt").input_ids
            if ids.shape[1] >= self.target_tokens:
                prompt = self.tokenizer.decode(
                    ids[0, : self.target_tokens],
                    skip_special_tokens=True,
                )
                prompt_id = f"{self.source_tag}_doc{offset}_{self.target_tokens}tok"
                return prompt, prompt_id

        raise ValueError(
            f"No document with >= {self.target_tokens} tokens found from idx={idx} in '{self.source_tag}'."
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
        prompt_id = f"random_vocab_seed{self.seed_base + idx}_{self.target_tokens}tok"
        return prompt, prompt_id
