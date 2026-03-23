"""
Data loading and text extraction utilities.
"""

from typing import List, Tuple

from datasets import load_dataset

from config import DATASET_NAME, DATASET_CONFIG


def load_wikitext(split: str = "train"):
    """Load WikiText-103 and return the requested split."""
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    return dataset[split]


def get_continuous_text(
    dataset_split,
    tokenizer,
    start_idx: int = 1000,
    num_tokens: int = 512,
) -> Tuple[str, int]:
    """
    Extract a continuous passage of *num_tokens* tokens from *dataset_split*,
    starting the scan at row *start_idx*. Empty lines and section headers
    (lines starting with '=') are skipped.

    Returns
    -------
    text : str
        The decoded text corresponding to exactly *num_tokens* tokens.
    actual_tokens : int
        The number of tokens (equals *num_tokens* unless the dataset is exhausted).
    """
    texts = []
    total_tokens = 0
    idx = start_idx

    while total_tokens < num_tokens and idx < len(dataset_split):
        text = dataset_split[idx]["text"].strip()
        if text and not text.startswith("="):
            toks = tokenizer(text, add_special_tokens=False)["input_ids"]
            total_tokens += len(toks)
            texts.append(text)
        idx += 1

    tokenized = tokenizer(
        " ".join(texts),
        return_tensors="pt",
        max_length=num_tokens,
        truncation=True,
        add_special_tokens=True,
    )

    decoded = tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=False)
    return decoded, tokenized["input_ids"].shape[1]


def get_prompt_batch(
    dataset_split,
    tokenizer,
    n_prompts: int = 50,
    num_tokens: int = 512,
    stride: int = 600,
    start_idx: int = 1000,
) -> List[Tuple[str, int]]:
    """
    Extract *n_prompts* non-overlapping text passages, each of *num_tokens* tokens.

    Each passage starts *stride* rows after the previous one in the dataset,
    ensuring textual independence between prompts. A stride > num_tokens (in
    approximate row-token terms) avoids any content overlap.

    Parameters
    ----------
    dataset_split :
        HuggingFace dataset split (e.g. from load_wikitext).
    tokenizer :
        HuggingFace tokenizer compatible with the target model.
    n_prompts : int
        Number of passages to extract.
    num_tokens : int
        Token length of each passage (after truncation).
    stride : int
        Row offset between the start of consecutive passages. Should be set
        larger than the average number of rows needed to fill *num_tokens*
        tokens, to guarantee non-overlapping content.
    start_idx : int
        Row index from which the first passage begins.

    Returns
    -------
    prompts : List[Tuple[str, int]]
        List of (decoded_text, actual_token_count) pairs.
    """
    prompts = []
    current_idx = start_idx

    for i in range(n_prompts):
        if current_idx >= len(dataset_split):
            raise ValueError(
                f"Dataset exhausted after {i} prompts "
                f"(last start_idx={current_idx}, dataset size={len(dataset_split)}). "
                f"Reduce n_prompts or stride."
            )
        text, n_toks = get_continuous_text(
            dataset_split, tokenizer,
            start_idx=current_idx,
            num_tokens=num_tokens,
        )
        prompts.append((text, n_toks))
        current_idx += stride

    return prompts
