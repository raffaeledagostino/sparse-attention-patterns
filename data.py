"""
Data loading and text extraction utilities.
"""

from datasets import load_dataset

from config import DATASET_NAME, DATASET_CONFIG


def load_wikitext(split: str = "train"):
    """Load WikiText-103 and return the requested split."""
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    return dataset[split]


def get_continuous_text(dataset_split, tokenizer, start_idx: int = 1000,
                        num_tokens: int = 512):
    """
    Extract a continuous passage of *num_tokens* tokens from *dataset_split*,
    starting the scan at row *start_idx*.  Empty lines and section headers
    (lines starting with '=') are skipped.

    Returns
    -------
    text : str
        The decoded text corresponding to exactly *num_tokens* tokens.
    actual_tokens : int
        The number of tokens (should equal *num_tokens* unless the dataset
        runs out of rows).
    """
    texts = []
    total_tokens = 0
    idx = start_idx

    while total_tokens < num_tokens and idx < len(dataset_split):
        text = dataset_split[idx]["text"].strip()

        # Skip empty lines and headers
        if text and not text.startswith("="):
            texts.append(text)

            combined_text = " ".join(texts)
            for row in dataset_split:
                text = row["text"].strip()
                if text and not text.startswith("="):
                    toks = tokenizer(text, add_special_tokens=False)["input_ids"]
                    total_tokens += len(toks)
                    texts.append(text)
                    if total_tokens >= num_tokens:
                        break

            total_tokens = tokenized["input_ids"].shape[1]

        idx += 1

    final_text = " ".join(texts)

    tokenized = tokenizer(
        final_text,
        return_tensors="pt",
        max_length=num_tokens,
        truncation=True,
        add_special_tokens=True,
    )

    final_text_truncated = tokenizer.decode(
        tokenized["input_ids"][0], skip_special_tokens=False
    )

    return final_text_truncated, tokenized["input_ids"].shape[1]
