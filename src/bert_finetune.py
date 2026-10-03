"""Fine-tune a pre-trained BERT / FinBERT model for 3-class tweet sentiment.

Usage (from the project root; a GPU is strongly recommended):
    python -m src.bert_finetune                               # bert-base-uncased
    python -m src.bert_finetune --model-name ProsusAI/finbert
    python -m src.bert_finetune --smoke-test                  # offline check

The same TRAIN -> (fit, dev) split as the RNN baselines is used: dev drives
early stopping and the official validation split is used only for the final
numbers, so the comparison with the baselines is like-for-like.

``--smoke-test`` builds a tiny randomly initialised BERT and a vocabulary from
the training tweets so the full loop can be exercised without internet
access. Its scores are meaningless and are saved under a separate name.
"""

import argparse
import copy
import json
import time

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BertConfig, BertForSequenceClassification,
                          BertTokenizerFast, get_linear_schedule_with_warmup)

from . import config
from .data import (URL_RE, Vocab, class_weights, load_splits, make_dev_split,
                   SPACE_RE)
from .evaluate import (compute_metrics, misclassified, plot_confusion,
                       save_metrics)
from .train import set_seed


def bert_clean(text):
    """Light cleaning for BERT: drop URLs only.

    BERT's WordPiece tokenizer handles casing, punctuation and rare words
    itself, so heavier cleaning would only remove signal.
    """
    return SPACE_RE.sub(" ", URL_RE.sub(" ", text)).strip()


class EncodedDataset(torch.utils.data.Dataset):
    """Pre-tokenised tensors for a list of texts."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer([bert_clean(t) for t in texts], truncation=True,
                             max_length=max_len, padding="max_length",
                             return_tensors="pt")
        self.labels = torch.tensor(list(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


def load_model_and_tokenizer(model_name):
    """Load a hub checkpoint with a fresh 3-way head and our label names."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=config.NUM_CLASSES, id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        # FinBERT ships its own (positive/negative/neutral) head in another
        # label order; re-initialise it instead of reusing the wrong mapping.
        ignore_mismatched_sizes=True)
    if "finbert" in model_name.lower():
        model.classifier.reset_parameters()
    return model, tokenizer


def smoke_model_and_tokenizer(train_texts, workdir):
    """Tiny random BERT + word-level vocab: exercises the code offline."""
    vocab = Vocab.build(train_texts, min_freq=3)
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = [w for w in vocab.itos[2:] if not w.startswith("$")]
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "vocab.txt").write_text("\n".join(specials + words))
    tokenizer = BertTokenizerFast(vocab_file=str(workdir / "vocab.txt"),
                                  do_lower_case=True)
    cfg = BertConfig(vocab_size=len(specials) + len(words), hidden_size=64,
                     num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, num_labels=config.NUM_CLASSES,
                     id2label=config.ID2LABEL, label2id=config.LABEL2ID)
    return BertForSequenceClassification(cfg), tokenizer


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for batch in loader:
        labels = batch.pop("labels")
        out = model(**{k: v.to(device) for k, v in batch.items()})
        probs.append(torch.softmax(out.logits, -1).cpu())
        ys.append(labels)
    probs = torch.cat(probs).numpy()
    return torch.cat(ys).numpy(), probs.argmax(1), probs


def main(args):
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_df, valid_df = load_splits()
    fit_df, dev_df = make_dev_split(train_df)
    if args.max_train:
        fit_df = fit_df.sample(min(args.max_train, len(fit_df)),
                               random_state=config.SEED)

    out_name = "BERT_smoke_test" if args.smoke_test else args.run_name
    out_dir = config.MODELS_DIR / out_name
    if args.smoke_test:
        model, tokenizer = smoke_model_and_tokenizer(fit_df["text"], out_dir)
        lr = 5e-4
    else:
        model, tokenizer = load_model_and_tokenizer(args.model_name)
        lr = args.lr
    model.to(device)
    print(f"device={device}  model={'smoke' if args.smoke_test else args.model_name}"
          f"  fit={len(fit_df)} dev={len(dev_df)} valid={len(valid_df)}")

    def loader(df, shuffle, bs):
        ds = EncodedDataset(df["text"], df["label"], tokenizer, args.max_len)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    train_loader = loader(fit_df, True, args.batch_size)
    dev_loader = loader(dev_df, False, 128)
    valid_loader = loader(valid_df, False, 128)

    # AdamW + linear warm-up/decay is the standard BERT fine-tuning recipe
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights(fit_df["label"]).to(device))

    best_f1, best_state, bad = -1.0, None, 0
    history = {"dev_f1": [], "train_loss": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total, n = time.time(), 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            out = model(**{k: v.to(device) for k, v in batch.items()})
            loss = loss_fn(out.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item() * len(labels)
            n += len(labels)
            if step % 50 == 0:
                print(f"    step {step}/{len(train_loader)} "
                      f"loss {total / n:.4f}", flush=True)
        y, p, _ = predict(model, dev_loader, device)
        dev_f1 = f1_score(y, p, average="macro")
        history["dev_f1"].append(dev_f1)
        history["train_loss"].append(total / n)
        print(f"  epoch {epoch}  train_loss {total / n:.4f}  "
              f"dev_macroF1 {dev_f1:.4f}  ({time.time() - start:.0f}s)",
              flush=True)
        if dev_f1 > best_f1:
            best_f1, bad = dev_f1, 0
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stopping")
                break

    model.load_state_dict(best_state)
    y_true, y_pred, probs = predict(model, valid_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    extra = {"family": "smoke" if args.smoke_test else "bert",
             "base_model": "tiny-random-bert" if args.smoke_test
             else args.model_name,
             "params": sum(p.numel() for p in model.parameters()),
             "train_size": len(fit_df), "history": history}
    save_metrics(out_name, metrics, extra)

    # Save model + tokenizer so the app / predict script can load them
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "config_project.json").write_text(json.dumps(
        {"kind": "bert", "max_len": args.max_len}))
    if not args.smoke_test:
        plot_confusion(metrics["confusion_matrix"], f"{out_name} (validation)",
                       config.FIGURES_DIR / f"cm_{out_name}.png")
        (config.METRICS_DIR / f"{out_name}_errors.jsonl").write_text("\n".join(
            json.dumps({"model": out_name, **r}) for r in misclassified(
                list(valid_df["text"]), y_true, y_pred, probs)))
    print(f"{out_name}: acc {metrics['accuracy']:.4f} "
          f"macroF1 {metrics['macro_f1']:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=config.BERT_MODEL_NAME)
    parser.add_argument("--run-name", default="BERT",
                        help="folder under models/ and metrics file name")
    parser.add_argument("--epochs", type=int, default=config.BERT_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.BERT_PATIENCE)
    parser.add_argument("--lr", type=float, default=config.BERT_LR)
    parser.add_argument("--batch-size", type=int,
                        default=config.BERT_BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=config.BERT_MAX_LEN)
    parser.add_argument("--max-train", type=int, default=None,
                        help="subsample the training set (CPU experiments)")
    parser.add_argument("--smoke-test", action="store_true")
    main(parser.parse_args())
