"""Automated inference: raw financial text in, sentiment out.

A paragraph is split into sentences, each sentence is scored, and the
paragraph label comes from the average of the sentence probabilities, so one
bearish line inside an otherwise neutral paragraph still shows up.

Usage (from the project root):
    python -m src.predict "Shares of $XYZ plunged after the guidance cut."
    python -m src.predict --file news.txt --model LSTM
    echo "Apple beats estimates." | python -m src.predict
"""

import argparse
import json
import re
import sys

import torch

from . import config

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z$\"'(])")


def split_sentences(paragraph):
    """Naive sentence splitter that keeps tickers like $AAPL intact."""
    parts = [s.strip() for s in SENT_SPLIT_RE.split(paragraph.strip())]
    return [s for s in parts if s]


def available_models():
    """Names of trained models found under ``models/`` (smoke test excluded)."""
    if not config.MODELS_DIR.exists():
        return []
    return sorted(p.name for p in config.MODELS_DIR.iterdir()
                  if p.is_dir() and not p.name.endswith("smoke_test")
                  and ((p / "model.pt").exists()
                       or (p / "config_project.json").exists()))


def best_model_name():
    """Pick the trained model with the highest validation macro F1."""
    names = available_models()
    if not names:
        raise SystemExit("No trained model found. Run "
                         "`python -m src.run_baselines` first.")
    scores = {}
    for name in names:
        path = config.METRICS_DIR / f"{name}.json"
        if path.exists():
            scores[name] = json.loads(path.read_text())["macro_f1"]
    return max(scores, key=scores.get) if scores else names[0]


class SentimentPredictor:
    """Uniform ``predict_proba`` over RNN baselines and fine-tuned BERT."""

    def __init__(self, name=None):
        self.name = name or best_model_name()
        path = config.MODELS_DIR / self.name
        if (path / "config_project.json").exists():
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)

            from .bert_finetune import bert_clean

            self.kind = "bert"
            self.max_len = json.loads(
                (path / "config_project.json").read_text())["max_len"]
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path).eval()
            self._clean = bert_clean
        else:
            from .train import load_baseline

            self.kind = "rnn"
            self.model, self.vocab = load_baseline(self.name)

    @torch.no_grad()
    def predict_proba(self, texts):
        """Return an (n, 3) tensor of class probabilities."""
        if self.kind == "bert":
            enc = self.tokenizer([self._clean(t) for t in texts],
                                 truncation=True, max_length=self.max_len,
                                 padding=True, return_tensors="pt")
            logits = self.model(**enc).logits
        else:
            ids = [self.vocab.encode(t) for t in texts]
            lengths = torch.tensor([len(i) for i in ids])
            padded = torch.full((len(ids), int(lengths.max())),
                                self.vocab.pad_idx, dtype=torch.long)
            for row, seq in enumerate(ids):
                padded[row, :len(seq)] = torch.tensor(seq)
            logits = self.model(padded, lengths)
        return torch.softmax(logits, dim=-1)

    def analyse(self, paragraph):
        """Sentence-level and aggregated paragraph-level sentiment."""
        sentences = split_sentences(paragraph) or [paragraph]
        probs = self.predict_proba(sentences)
        overall = probs.mean(dim=0)
        return {
            "model": self.name,
            "label": config.ID2LABEL[int(overall.argmax())],
            "probabilities": {config.ID2LABEL[i]: round(float(p), 4)
                              for i, p in enumerate(overall)},
            "sentences": [
                {"text": s, "label": config.ID2LABEL[int(p.argmax())],
                 "confidence": round(float(p.max()), 4)}
                for s, p in zip(sentences, probs)],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("text", nargs="?", help="paragraph to analyse")
    parser.add_argument("--file", help="read the paragraph from a text file")
    parser.add_argument("--model", help=f"one of {available_models()}; "
                        "default: best by validation macro F1")
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    args = parser.parse_args()

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()
    result = SentimentPredictor(args.model).analyse(text)

    if args.json:
        print(json.dumps(result, indent=2))
        return
    probs = "  ".join(f"{k} {v:.1%}" for k, v in
                      result["probabilities"].items())
    print(f"Model:     {result['model']}")
    print(f"Sentiment: {result['label'].upper()}   ({probs})")
    if len(result["sentences"]) > 1:
        print("Per sentence:")
        for s in result["sentences"]:
            print(f"  [{s['label']:<7} {s['confidence']:.0%}] {s['text']}")


if __name__ == "__main__":
    main()
