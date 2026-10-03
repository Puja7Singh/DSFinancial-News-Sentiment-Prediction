"""Train and evaluate every RNN-family baseline (plus a TF-IDF reference).

Usage (from the project root):
    python -m src.run_baselines
"""

import argparse
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import config
from .data import Vocab, class_weights, clean_text, load_splits, make_dev_split
from .evaluate import (compute_metrics, misclassified, plot_confusion,
                       save_metrics)
from .models import BASELINE_VARIANTS, RNNClassifier
from .train import (make_loader, predict_loader, save_baseline, set_seed,
                    train_model)


def run_tfidf(fit_df, valid_df):
    """Classical reference point: TF-IDF n-grams + logistic regression."""
    vec = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True)
    X = vec.fit_transform(fit_df["text"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0)
    clf.fit(X, fit_df["label"])
    pred = clf.predict(vec.transform(valid_df["text"]))
    return compute_metrics(valid_df["label"], pred)


def main(variants=None, epochs=config.MAX_EPOCHS):
    set_seed()
    train_df, valid_df = load_splits()
    fit_df, dev_df = make_dev_split(train_df)
    print(f"train {len(train_df)} -> fit {len(fit_df)} / dev {len(dev_df)}; "
          f"validation {len(valid_df)}")

    m = run_tfidf(fit_df, valid_df)
    save_metrics("TFIDF_LogReg", m, {"family": "classical"})
    print(f"TF-IDF + LogReg: acc {m['accuracy']:.4f} "
          f"macroF1 {m['macro_f1']:.4f}")

    vocab = Vocab.build(fit_df["text"])
    print(f"vocabulary size: {len(vocab)}")
    weights = class_weights(fit_df["label"])
    train_loader = make_loader(fit_df["text"], fit_df["label"], vocab, True)
    dev_loader = make_loader(dev_df["text"], dev_df["label"], vocab, False)
    valid_loader = make_loader(valid_df["text"], valid_df["label"], vocab,
                               False)

    errors = {}
    for name, kwargs in BASELINE_VARIANTS.items():
        if variants and name not in variants:
            continue
        print(f"\n=== {name} ===")
        set_seed()
        model = RNNClassifier(len(vocab), pad_idx=vocab.pad_idx, **kwargs)
        n_params = sum(p.numel() for p in model.parameters())
        model, hist = train_model(model, train_loader, dev_loader, weights,
                                  epochs=epochs)
        y_true, y_pred, probs = predict_loader(model, valid_loader)
        metrics = compute_metrics(y_true, y_pred)
        save_metrics(name, metrics, {"family": "rnn", "params": n_params,
                                     "history": hist})
        save_baseline(model, vocab, name, kwargs)
        plot_confusion(metrics["confusion_matrix"], f"{name} (validation)",
                       config.FIGURES_DIR / f"cm_{name}.png")
        errors[name] = misclassified(list(valid_df["text"]), y_true, y_pred,
                                     probs)
        print(f"{name}: acc {metrics['accuracy']:.4f} "
              f"macroF1 {metrics['macro_f1']:.4f}")

    (config.METRICS_DIR / "rnn_errors.jsonl").write_text(
        "\n".join(json.dumps({"model": k, **r})
                  for k, rows in errors.items() for r in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*",
                        choices=list(BASELINE_VARIANTS))
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    args = parser.parse_args()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    main(args.variants, args.epochs)
