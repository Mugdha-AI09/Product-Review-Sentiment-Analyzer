# ============================================================
# ⚙️  config.py  —  Shared settings for all pipeline stages
# ============================================================

import os

# ── Base directory (auto-detects wherever this folder is placed) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Input ─────────────────────────────────────────────────────
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")

# ── Intermediate artifacts (saved between pipeline stages) ────
PROCESSED_CSV  = os.path.join(BASE_DIR, "processed_data.csv")
TFIDF_MATRIX   = os.path.join(BASE_DIR, "tfidf_matrix.npz")
VECTORIZER_PKL = os.path.join(BASE_DIR, "vectorizer.pkl")
LABELS_PKL     = os.path.join(BASE_DIR, "labels.pkl")
MODELS_PKL     = os.path.join(BASE_DIR, "models.pkl")

# ── Output plots ──────────────────────────────────────────────
EDA_PLOT_PATH = os.path.join(BASE_DIR, "eda_plots.png")
CM_PLOT_PATH  = os.path.join(BASE_DIR, "confusion_matrix.png")

# ── TF-IDF settings ───────────────────────────────────────────
TFIDF_CONFIG = dict(
    ngram_range   = (1, 2),
    max_features  = 40_000,
    sublinear_tf  = True,
    min_df        = 2,
    max_df        = 0.92,
    strip_accents = "unicode",
    analyzer      = "word",
)

# ── Train / test split ────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
