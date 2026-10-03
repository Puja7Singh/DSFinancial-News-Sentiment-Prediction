"""Generate notebooks/financial_sentiment.ipynb from the project modules.

The notebook is a documented walk-through that calls the functions in
``src/`` so notebook and scripts never drift apart.
    python scripts/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/financial_sentiment.ipynb
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Financial News Sentiment Prediction using Deep Learning & BERT

Classify finance tweets as **Bearish**, **Bullish** or **Neutral**.

| Section | What happens |
|---|---|
| 1. Setup | imports, seeds, paths |
| 2. Data | load the official train / validation splits, basic stats |
| 3. Preprocessing | clean tweets, build vocabulary (train only), tensors |
| 4. RNN baselines | Embedding + SimpleRNN / LSTM / GRU / BiLSTM, early stopping |
| 5. Evaluation | accuracy, macro F1, class-wise precision/recall, confusion matrices, error analysis |
| 6. BERT (optional) | fine-tune `bert-base-uncased` or FinBERT on the same data |
| 7. Comparison | RNN vs BERT table + charts, PDF report |
| 8. Inference | raw paragraph in, sentiment out |

All reusable logic lives in `src/` (data, models, training, evaluation, inference);
this notebook calls those functions and explains each step.
""")

md("## 1. Setup")
code("""
# On Google Colab: upload/unzip the project, then
#   %cd sentiment-project
#   !pip install -r requirements.txt
import os, sys, json
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src import config
from src.data import (load_splits, make_dev_split, clean_text, tokenize,
                      Vocab, class_weights)
from src.models import RNNClassifier, BASELINE_VARIANTS
from src.train import (set_seed, make_loader, train_model, predict_loader,
                       save_baseline)
from src.evaluate import (compute_metrics, save_metrics, plot_confusion,
                          misclassified, load_all_metrics)

set_seed(config.SEED)
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", DEVICE)
""")

md("""
## 2. Load and understand the dataset

Dataset: [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment).
`load_splits()` reads `data/raw/sent_train.csv` and `sent_valid.csv` (the official
splits) and falls back to the Hugging Face Hub if they are missing.
Labels: `0 = Bearish`, `1 = Bullish`, `2 = Neutral`.
""")
code("""
train_df, valid_df = load_splits()
print(f"train: {len(train_df):,} tweets | validation: {len(valid_df):,} tweets")
train_df.head()
""")
code("""
dist = pd.DataFrame({
    "train": train_df["label"].value_counts().sort_index(),
    "validation": valid_df["label"].value_counts().sort_index(),
}).rename(index=config.ID2LABEL)
dist["train %"] = (dist["train"] / dist["train"].sum() * 100).round(1)
dist["validation %"] = (dist["validation"] / dist["validation"].sum() * 100).round(1)
display(dist)

lengths = train_df["text"].str.split().str.len()
print("words per tweet: mean %.1f, 95th pct %d, max %d"
      % (lengths.mean(), lengths.quantile(0.95), lengths.max()))
print("duplicate texts in train:", train_df["text"].duplicated().sum())
print("train/validation text overlap:",
      len(set(train_df["text"]) & set(valid_df["text"])))
""")
md("""
**Observations.** Neutral is ~65% of both splits, so accuracy alone is
misleading (always predicting Neutral scores ~65%). Macro F1, which weights the
three classes equally, is the main metric, and the loss uses class weights.
Tweets are short, so a 48-token limit truncates almost nothing.
""")
code("""
for label in range(3):
    print(f"--- {config.ID2LABEL[label]} ---")
    for t in train_df[train_df.label == label]["text"].sample(3, random_state=1):
        print("  ", t)
""")

md("""
## 3. Preprocessing

* `clean_text`: remove URLs and @mentions, unescape `&amp;`, collapse spaces,
  lower-case. **$tickers, numbers and percentages are kept** because they carry
  financial meaning ("-12%", "$TSLA").
* `tokenize`: regex tokeniser that keeps `$tsla` and `3.5%` as single tokens.
* **No leakage:** a stratified 10% *dev* slice of TRAIN is used for early
  stopping, and the vocabulary is built from the remaining training tweets only.
  The official validation split is used only for final evaluation.
""")
code("""
example = train_df["text"].iloc[3]
print("raw:     ", example)
print("cleaned: ", clean_text(example))
print("tokens:  ", tokenize(clean_text(example)))
""")
code("""
fit_df, dev_df = make_dev_split(train_df)
vocab = Vocab.build(fit_df["text"])
weights = class_weights(fit_df["label"])
print(f"fit {len(fit_df):,} | dev {len(dev_df):,} | validation {len(valid_df):,}")
print(f"vocabulary: {len(vocab):,} tokens (min freq {config.MIN_FREQ})")
print("class weights:", dict(zip(config.LABELS, weights.numpy().round(2))))

train_loader = make_loader(fit_df["text"], fit_df["label"], vocab, shuffle=True)
dev_loader = make_loader(dev_df["text"], dev_df["label"], vocab, shuffle=False)
valid_loader = make_loader(valid_df["text"], valid_df["label"], vocab, shuffle=False)

ids, lengths_, labels = next(iter(train_loader))
print("batch tensor:", tuple(ids.shape), "| lengths:", lengths_[:8].tolist())
unk = np.mean([t == vocab.unk_idx for s in valid_loader.dataset.ids for t in s])
print(f"validation tokens mapped to <unk>: {unk:.1%}")
""")

md("""
## 4. Baseline deep learning models (from scratch)

Every variant shares one architecture so the comparison isolates the recurrent cell:

```
token ids -> Embedding(128) -> RNN | LSTM | GRU (128 units, optional bidirectional)
          -> [final hidden state ; masked max-pool over time] -> Dropout(0.4) -> Linear(3)
```

* Sequences are **packed** so padding never reaches the RNN.
* The max-pool readout lets a single strong word anywhere in the tweet count.
* Training: Adam (lr 2e-3, halved when dev F1 plateaus), weighted cross-entropy,
  gradient clipping at 1.0, **early stopping** on dev macro F1 (patience 4),
  restoring the best epoch.
""")
code("""
results, histories, predictions = {}, {}, {}
for name, kwargs in BASELINE_VARIANTS.items():
    print(f"\\n=== {name} ===")
    set_seed(config.SEED)
    model = RNNClassifier(len(vocab), pad_idx=vocab.pad_idx, **kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    model, hist = train_model(model, train_loader, dev_loader, weights,
                              device=DEVICE, verbose=False)
    y_true, y_pred, probs = predict_loader(model, valid_loader, DEVICE)
    m = compute_metrics(y_true, y_pred)
    save_metrics(name, m, {"family": "rnn", "params": n_params, "history": hist})
    save_baseline(model.cpu(), vocab, name, kwargs)
    results[name], histories[name] = m, hist
    predictions[name] = (y_true, y_pred, probs)
    print(f"{name}: {n_params/1e6:.2f}M params | best epoch {hist['best_epoch']} | "
          f"val acc {m['accuracy']:.4f} | val macro F1 {m['macro_f1']:.4f}")
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for name, h in histories.items():
    axes[0].plot(h["train_loss"], label=f"{name} train")
    axes[0].plot(h["dev_loss"], "--", label=f"{name} dev")
    axes[1].plot(range(1, len(h["dev_f1"]) + 1), h["dev_f1"], label=name)
    axes[1].plot(h["best_epoch"], h["dev_f1"][h["best_epoch"] - 1], "ko", ms=4)
axes[0].set_title("Loss (solid = train, dashed = dev)"); axes[0].set_xlabel("epoch")
axes[1].set_title("Dev macro F1 (dot = epoch kept)"); axes[1].set_xlabel("epoch")
axes[0].legend(fontsize=6, ncol=2); axes[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
""")
md("""
Dev loss starts rising after 3-5 epochs while dev F1 keeps creeping up: the
models become over-confident on the training data. Early stopping on dev F1
keeps the best-generalising epoch; the gap is the main reason the RNNs need
regularisation (dropout, small hidden size) on a dataset of this size.
""")

md("## 5. Evaluation on the validation split")
code("""
# Reference point: TF-IDF + logistic regression (classical, not deep learning)
from src.run_baselines import run_tfidf
tfidf = run_tfidf(fit_df, valid_df)
save_metrics("TFIDF_LogReg", tfidf, {"family": "classical"})

rows = []
for name, m in {"TFIDF_LogReg": tfidf, **results}.items():
    row = {"model": name, "accuracy": m["accuracy"], "macro F1": m["macro_f1"]}
    for c in config.LABELS:
        row[f"{c} P"] = m["per_class"][c]["precision"]
        row[f"{c} R"] = m["per_class"][c]["recall"]
    rows.append(row)
table = pd.DataFrame(rows).set_index("model").round(3)
table.sort_values("macro F1", ascending=False)
""")
code("""
best_rnn = max(results, key=lambda n: results[n]["macro_f1"])
fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.4))
for ax, (name, m) in zip(axes, results.items()):
    plot_confusion(m["confusion_matrix"], name, ax=ax)
plt.tight_layout(); plt.show()
print("best RNN variant:", best_rnn)
""")
md("""
**Which RNN variant works best and why.** The gated cells (LSTM, GRU) beat the
vanilla RNN: their gates stop a sentiment word early in a tweet from being
washed out by later tickers, source tags and URLs, and they train more stably.
GRU matches or beats LSTM with fewer parameters, which suits ~8.6k training
tweets. The BiLSTM doubles capacity and mostly memorises. Note that TF-IDF +
logistic regression is ahead of all from-scratch RNNs: with short headlines and
little data, randomly initialised embeddings cannot learn rare sentiment words
well. That is the gap pre-trained transformers are designed to close.
""")
md("### Error analysis: the best RNN's most confident mistakes")
code("""
y_true, y_pred, probs = predictions[best_rnn]
errors = pd.DataFrame(misclassified(list(valid_df["text"]), y_true, y_pred, probs, n=12))
pd.set_option("display.max_colwidth", 140)
errors
""")
code("""
cm = np.array(results[best_rnn]["confusion_matrix"])
share = cm / cm.sum(1, keepdims=True)
print(f"Bearish predicted Neutral: {share[0, 2]:.0%} | Bullish predicted Neutral: {share[1, 2]:.0%}")
print(f"Bearish <-> Bullish swaps: {share[0, 1]:.0%} / {share[1, 0]:.0%}")
""")
md("""
**Interpretation.**
1. Most errors are *missed* sentiment (Bearish/Bullish predicted Neutral), not
   polarity flips.
2. **Implicit sentiment**: no polar keyword, the meaning needs world knowledge
   ("TfL decided not to extend Uber's license").
3. **Domain semantics**: "dumping virus hedges" is bullish even though "dumping"
   usually signals selling pressure.
4. **Quotes and hedged forecasts** ("the rebound can be robust") read as neutral.
5. Some labels are borderline (press-release style headlines labelled Bearish).

Pre-trained language models address 2-4 directly, which motivates section 6.
""")

md("""
## 6. (Optional) Fine-tune BERT / FinBERT

`src/bert_finetune.py` fine-tunes a pre-trained checkpoint with the **same
fit / dev / validation split**:

* light cleaning (URLs removed; casing and punctuation left to WordPiece),
* max length 64, batch 32, AdamW lr 2e-5, weight decay 0.01, linear warm-up/decay,
* class-weighted loss, early stopping on dev macro F1 (patience 2, max 4 epochs).

A GPU is strongly recommended (Colab T4: ~2-3 min per epoch). Set
`BERT_MODEL = "ProsusAI/finbert"` to start from a finance-domain checkpoint.
If the Hub is unreachable the cell explains how to run it elsewhere.
""")
code("""
import argparse
from src import bert_finetune

BERT_MODEL = config.BERT_MODEL_NAME     # or "ProsusAI/finbert"
RUN_BERT = torch.cuda.is_available()    # set True to force a (slow) CPU run

bert_args = argparse.Namespace(
    model_name=BERT_MODEL, run_name="BERT", epochs=config.BERT_EPOCHS,
    patience=config.BERT_PATIENCE, lr=config.BERT_LR,
    batch_size=config.BERT_BATCH_SIZE, max_len=config.BERT_MAX_LEN,
    max_train=None, smoke_test=False)

if RUN_BERT:
    try:
        bert_metrics = bert_finetune.main(bert_args)
    except OSError as err:   # no internet access to the Hugging Face Hub
        print("Could not download", BERT_MODEL, "->", err)
elif (config.METRICS_DIR / "BERT.json").exists():
    print("Using saved BERT results from reports/metrics/BERT.json")
else:
    print("No GPU here: skipping BERT. Run this notebook on Colab (GPU runtime) "
          "or `python -m src.bert_finetune` to add BERT to the comparison.")
""")
code("""
# Offline check of the BERT pipeline: a tiny random BERT trained for 1 epoch.
# It only proves the code path works; its scores are NOT a BERT result.
smoke = argparse.Namespace(**{**vars(bert_args), "smoke_test": True,
                              "epochs": 1, "max_train": 1000})
_ = bert_finetune.main(smoke)
""")

md("## 7. RNN vs BERT comparison")
code("""
all_metrics = {k: v for k, v in load_all_metrics().items()
               if v.get("family") != "smoke"}
comp = pd.DataFrame({k: {"accuracy": v["accuracy"], "macro F1": v["macro_f1"],
                         **{f"{c} F1": v["per_class"][c]["f1-score"] for c in config.LABELS}}
                     for k, v in all_metrics.items()}).T.round(3)
comp = comp.sort_values("macro F1", ascending=False)
display(comp)

best = comp.loc[[best_rnn]].iloc[0]
if "BERT" in comp.index:
    gain = comp.loc["BERT", "macro F1"] - best["macro F1"]
    print(f"BERT improves macro F1 over {best_rnn} by {gain:+.3f} "
          f"({gain / best['macro F1']:+.1%})")
else:
    print("BERT not trained in this run (see section 6).")

ax = comp[["accuracy", "macro F1"]].plot.bar(figsize=(8, 3), ylim=(0.5, 1), rot=0)
ax.set_title("Validation scores by model"); plt.tight_layout(); plt.show()
""")
code("""
# Regenerate the 2-page PDF comparison report from the saved metrics
!python scripts/make_report.py
""")

md("""
## 8. Automated inference: raw paragraph -> sentiment

`SentimentPredictor` loads the best saved model (BERT if trained, otherwise the
best RNN by validation macro F1), splits a paragraph into sentences, scores each
one and averages the probabilities. The same is available from the command line:

```bash
python -m src.predict "Shares of $XYZ plunged after the guidance cut."
```
""")
code("""
from src.predict import SentimentPredictor

predictor = SentimentPredictor()          # or SentimentPredictor("LSTM")
paragraph = ("Oil prices slide as OPEC+ signals higher output. Energy names lead "
             "the S&P 500 lower. Apple beats on revenue and raises its dividend.")
result = predictor.analyse(paragraph)
print("model:", result["model"], "| paragraph sentiment:", result["label"])
print(result["probabilities"])
pd.DataFrame(result["sentences"])
""")
code("""
for text in ["$NFLX surges after subscriber numbers beat expectations",
             "Boeing shares tumble as FAA grounds more jets",
             "Fed minutes due at 2pm ET",
             "Tesla cut to Sell at Goldman Sachs"]:
    r = predictor.analyse(text)
    print(f"{r['label']:<8} {max(r['probabilities'].values()):.0%}  {text}")
""")
md("""
## Summary

* Leakage-free pipeline: dev slice from TRAIN for early stopping, vocabulary
  from training tweets only, official validation split for all reported numbers.
* Four from-scratch recurrent models compared; gated cells (GRU/LSTM) win.
* BERT fine-tuning code is included and runs on the same splits; its numbers
  are added to the comparison and PDF report automatically once trained.
* Streamlit dashboard: `streamlit run app/streamlit_app.py`.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                "language": "python"}
out = ROOT / "notebooks" / "financial_sentiment.ipynb"
nbf.write(nb, out)
print("wrote", out)
