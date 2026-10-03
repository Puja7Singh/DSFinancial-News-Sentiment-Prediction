"""Central configuration: paths, label names and default hyperparameters."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (everything is relative to the project root so the repo is portable)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_CSV = RAW_DIR / "sent_train.csv"
VALID_CSV = RAW_DIR / "sent_valid.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Hugging Face dataset id (used when the CSVs are not present locally)
HF_DATASET = "zeroshot/twitter-financial-news-sentiment"

# ---------------------------------------------------------------------------
# Labels (dataset card: LABEL_0 Bearish, LABEL_1 Bullish, LABEL_2 Neutral)
# ---------------------------------------------------------------------------
LABELS = ["Bearish", "Bullish", "Neutral"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for i, name in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# ---------------------------------------------------------------------------
# Baseline (RNN family) hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
DEV_FRACTION = 0.1      # slice of TRAIN held out for early stopping
MIN_FREQ = 2            # minimum token frequency to enter the vocabulary
MAX_LEN = 48            # tweets are short; 48 tokens covers >99% of them
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.4
BATCH_SIZE = 64
LEARNING_RATE = 2e-3
MAX_EPOCHS = 25
PATIENCE = 4            # early-stopping patience (epochs without dev F1 gain)

# ---------------------------------------------------------------------------
# BERT fine-tuning defaults
# ---------------------------------------------------------------------------
BERT_MODEL_NAME = "bert-base-uncased"   # or "ProsusAI/finbert"
BERT_MAX_LEN = 64
BERT_BATCH_SIZE = 32
BERT_LR = 2e-5
BERT_EPOCHS = 4
BERT_PATIENCE = 2
