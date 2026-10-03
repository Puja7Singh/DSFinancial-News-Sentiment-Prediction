"""Data loading, tweet cleaning, vocabulary building and PyTorch datasets.

The pipeline is deliberately leakage-free:
  * the vocabulary is built from the training portion only;
  * early stopping uses a dev slice carved out of TRAIN, so the official
    validation split is touched only for final evaluation.
"""

import re
from collections import Counter

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from . import config

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_splits():
    """Return (train_df, valid_df) with columns ``text`` and ``label``.

    Reads the local CSVs when present, otherwise downloads the dataset from
    the Hugging Face Hub.
    """
    if config.TRAIN_CSV.exists() and config.VALID_CSV.exists():
        train_df = pd.read_csv(config.TRAIN_CSV)
        valid_df = pd.read_csv(config.VALID_CSV)
    else:
        from datasets import load_dataset

        ds = load_dataset(config.HF_DATASET)
        train_df = ds["train"].to_pandas()
        valid_df = ds["validation"].to_pandas()
    for df in (train_df, valid_df):
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
    return train_df, valid_df


def make_dev_split(train_df, fraction=config.DEV_FRACTION, seed=config.SEED):
    """Stratified split of TRAIN into (fit_df, dev_df) for early stopping."""
    fit_df, dev_df = train_test_split(
        train_df, test_size=fraction, stratify=train_df["label"],
        random_state=seed)
    return fit_df.reset_index(drop=True), dev_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cleaning and tokenisation
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")
# Tokens: $tickers, numbers with optional %/decimals, words, or single symbols
TOKEN_RE = re.compile(r"\$[a-z][a-z.]*|\d+(?:[.,]\d+)*%?|[a-z]+(?:'[a-z]+)?|[!?%]")


def clean_text(text):
    """Remove URLs, @mentions and extra whitespace; keep $tickers and numbers.

    Lower-casing is applied so "Beats" and "beats" share one embedding.
    """
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("&amp;", "&")
    return SPACE_RE.sub(" ", text).strip().lower()


def tokenize(text):
    """Split a cleaned tweet into tokens (tickers are kept as one token)."""
    return TOKEN_RE.findall(text)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PAD, UNK = "<pad>", "<unk>"


class Vocab:
    """Token <-> index mapping built from training text only."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_idx = self.stoi[PAD]
        self.unk_idx = self.stoi[UNK]

    @classmethod
    def build(cls, texts, min_freq=config.MIN_FREQ):
        counts = Counter(tok for t in texts for tok in tokenize(clean_text(t)))
        itos = [PAD, UNK] + sorted(
            tok for tok, c in counts.items() if c >= min_freq)
        return cls(itos)

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len=config.MAX_LEN):
        """Clean, tokenise and map to ids (truncated, *not* padded)."""
        ids = [self.stoi.get(tok, self.unk_idx)
               for tok in tokenize(clean_text(text))]
        return ids[:max_len] or [self.unk_idx]


# ---------------------------------------------------------------------------
# PyTorch dataset + collate
# ---------------------------------------------------------------------------


class TweetDataset(Dataset):
    """Holds encoded tweets and labels."""

    def __init__(self, texts, labels, vocab, max_len=config.MAX_LEN):
        self.ids = [vocab.encode(t, max_len) for t in texts]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


def make_collate(pad_idx):
    """Pad a batch to its longest sequence and return lengths for packing."""

    def collate(batch):
        seqs, labels = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs])
        padded = torch.full((len(seqs), int(lengths.max())), pad_idx,
                            dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = torch.tensor(s)
        return padded, lengths, torch.tensor(labels)

    return collate


def class_weights(labels, num_classes=config.NUM_CLASSES):
    """Inverse-frequency class weights to counter the Neutral majority."""
    counts = torch.bincount(torch.tensor(list(labels)), minlength=num_classes)
    weights = counts.sum() / (num_classes * counts.float())
    return weights
