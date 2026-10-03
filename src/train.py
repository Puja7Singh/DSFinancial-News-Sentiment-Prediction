"""Training loop with early stopping for the RNN-family baselines."""

import copy
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from . import config
from .data import TweetDataset, Vocab, make_collate
from .models import RNNClassifier


def set_seed(seed=config.SEED):
    """Make runs repeatable (CPU-deterministic; GPU close to it)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(texts, labels, vocab, shuffle):
    ds = TweetDataset(texts, labels, vocab)
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle,
                      collate_fn=make_collate(vocab.pad_idx))


@torch.no_grad()
def predict_loader(model, loader, device="cpu"):
    """Return (y_true, y_pred, probs) as numpy arrays."""
    model.eval()
    ys, probs = [], []
    for ids, lengths, labels in loader:
        logits = model(ids.to(device), lengths)
        probs.append(torch.softmax(logits, dim=-1).cpu())
        ys.append(labels)
    probs = torch.cat(probs).numpy()
    y_true = torch.cat(ys).numpy()
    return y_true, probs.argmax(1), probs


def train_model(model, train_loader, dev_loader, weights=None,
                epochs=config.MAX_EPOCHS, patience=config.PATIENCE,
                lr=config.LEARNING_RATE, device="cpu", verbose=True):
    """Train with Adam + weighted cross-entropy; early stop on dev macro F1.

    Returns the model restored to its best epoch and a history dict.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1)
    loss_fn = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None)
    history = {"train_loss": [], "dev_f1": [], "dev_loss": []}
    best_f1, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        start, total, n = time.time(), 0.0, 0
        for ids, lengths, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(ids, lengths), labels)
            loss.backward()
            # Clipping keeps the vanilla RNN from exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
            n += len(labels)

        # Dev evaluation (dev is a slice of TRAIN, not the validation split)
        model.eval()
        dev_loss, dn, preds, ys = 0.0, 0, [], []
        with torch.no_grad():
            for ids, lengths, labels in dev_loader:
                logits = model(ids.to(device), lengths)
                dev_loss += loss_fn(logits, labels.to(device)).item() \
                    * len(labels)
                dn += len(labels)
                preds.append(logits.argmax(1).cpu())
                ys.append(labels)
        dev_f1 = f1_score(torch.cat(ys), torch.cat(preds), average="macro")
        scheduler.step(dev_f1)
        history["train_loss"].append(total / n)
        history["dev_loss"].append(dev_loss / dn)
        history["dev_f1"].append(dev_f1)
        if verbose:
            print(f"  epoch {epoch:2d}  train_loss {total / n:.4f}  "
                  f"dev_loss {dev_loss / dn:.4f}  dev_macroF1 {dev_f1:.4f}  "
                  f"({time.time() - start:.1f}s)")

        if dev_f1 > best_f1:
            best_f1, bad_epochs = dev_f1, 0
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"  early stopping (best epoch "
                          f"{history['best_epoch']})")
                break

    model.load_state_dict(best_state)
    history["best_dev_f1"] = best_f1
    return model, history


# ---------------------------------------------------------------------------
# Save / load so the Streamlit app and predict script can reuse models
# ---------------------------------------------------------------------------


def save_baseline(model, vocab, name, model_kwargs):
    path = config.MODELS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "vocab.json").write_text(json.dumps(vocab.itos))
    (path / "config.json").write_text(json.dumps(
        {"kind": "rnn", "model_kwargs": model_kwargs,
         "max_len": config.MAX_LEN}, indent=2))
    return path


def load_baseline(name):
    path = config.MODELS_DIR / name
    vocab = Vocab(json.loads((path / "vocab.json").read_text()))
    cfg = json.loads((path / "config.json").read_text())
    model = RNNClassifier(len(vocab), pad_idx=vocab.pad_idx,
                          **cfg["model_kwargs"])
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
    model.eval()
    return model, vocab
