# ============================================================
# 📦 Import Libraries
# ============================================================
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# Download NLTK assets (only first run)
nltk.download("stopwords",        quiet=True)
nltk.download("wordnet",          quiet=True)
nltk.download("averaged_perceptron_tagger",     quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt",            quiet=True)
nltk.download("punkt_tab",        quiet=True)

# ============================================================
# 📂 Load Dataset
# ============================================================
df = pd.read_csv(r"C:\Users\Bhumi\OneDrive\Desktop\sentiment_analysis\dataset.csv")
df = df[['cleaned_review', 'sentiment_category']].dropna()

# ============================================================
# 🧹 Enhanced Text Preprocessing  (Lemmatizer — gentler than Stemmer)
# ============================================================
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

# Keep negation words — critical for sentiment
negation_words = {"no", "not", "nor", "never", "neither", "without",
                  "nobody", "nothing", "nowhere", "hardly", "barely",
                  "doesn't", "isn't", "wasn't", "wouldn't", "couldn't"}
stop_words -= negation_words

def _get_wordnet_pos(treebank_tag: str):
    """Map POS tag to WordNet POS for accurate lemmatization."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN   # default

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<.*?>",         " ", text)   # strip HTML
    text = re.sub(r"http\S+|www\S+", " ", text)  # strip URLs
    text = re.sub(r"[^a-z\s]",      " ", text)   # keep only letters
    # Use simple split (10x faster than word_tokenize for 21k rows)
    # Batch POS tag the whole list at once instead of per-token
    tokens   = [t for t in text.split() if t not in stop_words and len(t) > 2]
    pos_tags = pos_tag(tokens)                    # single batch call per review
    tokens   = [
        lemmatizer.lemmatize(tok, _get_wordnet_pos(pos))
        for tok, pos in pos_tags
    ]
    return " ".join(tokens)

print("🔄  Preprocessing text ...")
# tqdm gives a live progress bar so you can see it's working
from tqdm import tqdm
tqdm.pandas(desc="  Processing rows")
df['processed_review'] = df['cleaned_review'].progress_apply(preprocess)
print("Preprocessing done ✅")

# ============================================================
# 🔤 Encode Labels
# ============================================================
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment_category'])

# ============================================================
# 📊 EDA — Plot 1 : Class Distribution
#          Plot 2 : Top-20 Words per Sentiment
# ============================================================
print("📊  Generating EDA plots ...")

COLORS = {
    "Positive" : "#2ecc71",
    "Neutral"  : "#3498db",
    "Negative" : "#e74c3c",
}
CLASS_ORDER = ["Positive", "Neutral", "Negative"]

fig = plt.figure(figsize=(18, 7))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# ── Plot 1 : Sentiment Distribution ──────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor("#0d1117")

counts    = df['sentiment_category'].value_counts()
cats      = [c for c in CLASS_ORDER if c in counts.index]
vals      = [counts[c] for c in cats]
bar_cols  = [COLORS[c] for c in cats]
pct_total = sum(vals)

bars = ax1.bar(cats, vals, color=bar_cols, width=0.55,
               edgecolor="#0d1117", linewidth=1.5,
               zorder=3)

for bar, v in zip(bars, vals):
    pct = v / pct_total * 100
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + pct_total * 0.01,
             f"{v:,}\n({pct:.1f}%)",
             ha="center", va="bottom",
             color="white", fontsize=11, fontweight="bold")

ax1.set_title("Sentiment Class Distribution",
              color="white", fontsize=14, fontweight="bold", pad=12)
ax1.set_xlabel("Sentiment", color="#aaaaaa", fontsize=11)
ax1.set_ylabel("Count",     color="#aaaaaa", fontsize=11)
ax1.tick_params(colors="white")
ax1.spines[["top","right","left","bottom"]].set_color("#333333")
ax1.yaxis.grid(True, color="#222222", linestyle="--", linewidth=0.7, zorder=0)
ax1.set_axisbelow(True)

# ── Plot 2 : Top-20 Words per Sentiment ──────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#0d1117")

TOP_N = 20
word_freq: dict[str, Counter] = {}
for cls in CLASS_ORDER:
    subset = df[df['sentiment_category'] == cls]['processed_review']
    all_words = " ".join(subset).split()
    word_freq[cls] = Counter(all_words).most_common(TOP_N)

# Stacked horizontal grouped display
y_positions = np.arange(TOP_N)
bar_height  = 0.25
offsets     = [-bar_height, 0, bar_height]

for i, cls in enumerate(CLASS_ORDER):
    if cls not in word_freq:
        continue
    words  = [w for w, _ in word_freq[cls]][::-1]
    freqs  = [c for _, c in word_freq[cls]][::-1]
    y_pos  = y_positions + offsets[i]
    ax2.barh(y_pos, freqs, height=bar_height * 0.85,
             color=COLORS[cls], label=cls, alpha=0.9, zorder=3)

# Use words from Positive class for y-axis labels (most interpretable)
ref_words = [w for w, _ in word_freq.get("Positive", word_freq[CLASS_ORDER[0]])][::-1]
ax2.set_yticks(y_positions)
ax2.set_yticklabels(ref_words, color="white", fontsize=8)
ax2.tick_params(axis="x", colors="white")
ax2.set_title("Top 20 Words per Sentiment Class",
              color="white", fontsize=14, fontweight="bold", pad=12)
ax2.set_xlabel("Frequency", color="#aaaaaa", fontsize=11)
ax2.spines[["top","right","left","bottom"]].set_color("#333333")
ax2.xaxis.grid(True, color="#222222", linestyle="--", linewidth=0.7, zorder=0)
ax2.set_axisbelow(True)
ax2.legend(facecolor="#1a1a2e", labelcolor="white",
           framealpha=0.8, fontsize=10, loc="lower right")

fig.suptitle("🔍  Exploratory Data Analysis  —  Sentiment Dataset",
             color="white", fontsize=16, fontweight="bold", y=1.01)
plt.savefig(r"C:\Users\Bhumi\OneDrive\Desktop\sentiment_analysis\eda_plots.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show(block=False)
plt.pause(1)
plt.close()
print("   Saved → eda_plots.png ✅\n")

# ============================================================
# 📝 TF-IDF with N-grams (unigrams + bigrams)
#    Tuned for 21k rows
# ============================================================
vectorizer = TfidfVectorizer(
    ngram_range   = (1, 2),      # unigrams + bigrams
    max_features  = 40_000,      # slightly more than 50k to capture key bigrams
    sublinear_tf  = True,        # log normalization — proven to help SVM/LR
    min_df        = 2,           # lowered: include terms in ≥2 docs
    max_df        = 0.92,        # ignore terms in >92% of docs
    strip_accents = "unicode",
    analyzer      = "word",
)

X = vectorizer.fit_transform(df['processed_review'])
y = df['sentiment']

# ============================================================
# 📊 Dataset Summary
# ============================================================
print("=" * 56)
print("            📊  DATASET SUMMARY")
print("=" * 56)
print(f"  Total samples      : {len(df):,}")
print(f"  Shape of X         : {X.shape}")
print(f"  Sample labels      : {list(y[:5])}")
print()
print("  Classes Mapping:")
for label in le.classes_:
    idx = le.transform([label])[0]
    cnt = (df['sentiment_category'] == label).sum()
    print(f"    {label:>10}  →  {idx}   ({cnt:,} samples)")

# ============================================================
# ✂️ Train-Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print()
print("  Train-Test Split:")
print(f"    Training set   : {X_train.shape}")
print(f"    Testing set    : {X_test.shape}")
print("=" * 56)

# ============================================================
# 🔧 Hyperparameter Grids
# ============================================================
from sklearn.linear_model import SGDClassifier

param_grids = {
    "Logistic Regression": {
        # class_weight='balanced' helps if any sentiment class is slightly uneven
        # saga solver handles large sparse TF-IDF matrices best
       "estimator" : LogisticRegression(max_iter=1000, solver="lbfgs",
                                  class_weight="balanced",
                                  multi_class="auto"),
       "param_grid": {
                 "C": [0.1, 1, 5, 10],
                     },
    },
    "Naive Bayes": {
        "estimator" : MultinomialNB(),
        "param_grid": {"alpha": [0.02, 0.05, 0.1, 0.2, 0.5]},
    },
    "SVM (LinearSVC)": {
        # class_weight='balanced' + finer C search around 1–10 sweet spot
        "estimator" : LinearSVC(max_iter=3000, dual=True,
                                class_weight="balanced"),
        "param_grid": {"C": [0.5, 1, 2, 5, 10, 20]},
    },
    "SGD Classifier": {
        # SGD with hinge loss ≈ linear SVM but with more regularization options
        # log_loss gives predict_proba support for free (useful for ensemble)
        "estimator" : SGDClassifier(loss="modified_huber", max_iter=1000,
                                    class_weight="balanced", random_state=42),
        "param_grid": {
            "alpha"  : [1e-5, 1e-4, 1e-3, 1e-2],
            "penalty": ["l2", "elasticnet"],
        },
    },
}

# ============================================================
# 🤖 Train Individual Models with GridSearchCV
# ============================================================
print()
print("=" * 60)
print("   🔧  HYPERPARAMETER TUNING  (GridSearchCV, cv=5)")
print("=" * 60)

results    = {}
best_ests  = {}   # store best estimators for ensemble

for name, cfg in param_grids.items():
    print(f"\n  Tuning {name} ...", end=" ", flush=True)
    gs_cv = GridSearchCV(
    cfg["estimator"], cfg["param_grid"],
    cv=3, scoring="f1_weighted", n_jobs=-1
    )
    gs_cv.fit(X_train, y_train)
    best_est        = gs_cv.best_estimator_
    best_ests[name] = best_est
    y_pred          = best_est.predict(X_test)

    acc       = accuracy_score(y_test, y_pred) * 100
    f1_w      = f1_score(y_test, y_pred, average="weighted") * 100
    f1_m      = f1_score(y_test, y_pred, average="macro")    * 100
    precision = precision_score(y_test, y_pred, average="weighted") * 100
    recall    = recall_score(y_test, y_pred, average="weighted")    * 100

    results[name] = {
        "model"       : best_est,
        "best_params" : gs_cv.best_params_,
        "accuracy"    : acc,
        "f1_weighted" : f1_w,
        "f1_macro"    : f1_m,
        "precision"   : precision,
        "recall"      : recall,
        "y_pred"      : y_pred,
        "report"      : classification_report(y_test, y_pred,
                                               target_names=le.classes_),
    }
    print(f"done ✅  best params: {gs_cv.best_params_}")

# ============================================================
# 🗳️  Voting Ensemble  (soft voting across all best models)
#     LinearSVC doesn't support predict_proba natively →
#     wrap with CalibratedClassifierCV (cv=3) so
#     VotingClassifier can refit it cleanly during ensemble.fit()
#     SGDClassifier with modified_huber already supports predict_proba ✅
# ============================================================
print(f"\n  Building Voting Ensemble ...", end=" ", flush=True)

best_svm_C = best_ests["SVM (LinearSVC)"].C

calibrated_svm = CalibratedClassifierCV(
    LinearSVC(C=best_svm_C, max_iter=3000, dual=True,
              class_weight="balanced"),
    cv=3
)

ensemble = VotingClassifier(
    estimators=[
        ("lr",  best_ests["Logistic Regression"]),
        ("nb",  best_ests["Naive Bayes"]),
        ("svm", calibrated_svm),
        ("sgd", best_ests["SGD Classifier"]),
    ],
    voting="soft",   # uses predicted probabilities
    n_jobs=1,        # avoid joblib subprocess issues on Windows
)
ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_test)

acc_e       = accuracy_score(y_test, y_pred_ens) * 100
f1_w_e      = f1_score(y_test, y_pred_ens, average="weighted") * 100
f1_m_e      = f1_score(y_test, y_pred_ens, average="macro")    * 100
precision_e = precision_score(y_test, y_pred_ens, average="weighted") * 100
recall_e    = recall_score(y_test, y_pred_ens, average="weighted")    * 100

results["Voting Ensemble"] = {
    "model"       : ensemble,
    "best_params" : {"voting": "soft", "models": "LR + NB + SVM"},
    "accuracy"    : acc_e,
    "f1_weighted" : f1_w_e,
    "f1_macro"    : f1_m_e,
    "precision"   : precision_e,
    "recall"      : recall_e,
    "y_pred"      : y_pred_ens,
    "report"      : classification_report(y_test, y_pred_ens,
                                           target_names=le.classes_),
}
print(f"done ✅")

# ============================================================
# 📈 Metrics Comparison Table
# ============================================================
print()
print("=" * 72)
print("                    📈  MODEL METRICS COMPARISON")
print("=" * 72)

COL_W = 20
header = (f"  {'Model':<{COL_W}} {'Accuracy':>10} {'F1-Weighted':>13}"
          f" {'F1-Macro':>10} {'Precision':>11} {'Recall':>8}")
print(header)
print("  " + "-" * 76)

for name, res in results.items():
    print(f"  {name:<{COL_W}}"
          f" {res['accuracy']:>9.2f}%"
          f" {res['f1_weighted']:>12.2f}%"
          f" {res['f1_macro']:>9.2f}%"
          f" {res['precision']:>10.2f}%"
          f" {res['recall']:>7.2f}%")

print("  " + "-" * 76)
print()

# ── Visual accuracy bars ──────────────────────────────────────
for name, res in results.items():
    bar_len = int(res["accuracy"] / 2)
    bar     = "█" * bar_len + "░" * (50 - bar_len)
    print(f"  {name}")
    print(f"  {res['accuracy']:.2f}%  [{bar}]")
    print()

# ============================================================
# 🏆 Best Model (by F1-Weighted)
# ============================================================
best_name  = max(results, key=lambda k: results[k]["f1_weighted"])
best       = results[best_name]
best_model = best["model"]

print("=" * 72)
print("                        🏆  BEST MODEL")
print("=" * 72)
print(f"  ✅  {best_name}")
print(f"      Best params  : {best['best_params']}")
print(f"      Accuracy     : {best['accuracy']:.2f}%")
print(f"      F1-Weighted  : {best['f1_weighted']:.2f}%   ← used for model selection")
print(f"      F1-Macro     : {best['f1_macro']:.2f}%")
print(f"      Precision    : {best['precision']:.2f}%")
print(f"      Recall       : {best['recall']:.2f}%")
print()
print("  Classification Report:")
print(best["report"])
print("=" * 72)

# ============================================================
# 📉 Confusion Matrix for Best Model
# ============================================================
fig2, ax = plt.subplots(figsize=(7, 6))
fig2.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

cm  = confusion_matrix(y_test, best["y_pred"])
cmd = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=le.classes_)
cmd.plot(ax=ax, colorbar=True, cmap="Blues")

ax.set_title(f"Confusion Matrix — {best_name}",
             color="white", fontsize=13, fontweight="bold", pad=10)
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
plt.setp(ax.get_xticklabels(), color="white")
plt.setp(ax.get_yticklabels(), color="white")

plt.tight_layout()
plt.savefig(r"C:\Users\Bhumi\OneDrive\Desktop\sentiment_analysis\confusion_matrix.png",
            dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.show(block=False)
plt.pause(1)
plt.close()
print("   Confusion matrix saved → confusion_matrix.png ✅\n")

# ============================================================
# 💬 User Input Prediction Loop
# ============================================================
print()
print("=" * 56)
print("   💬  SENTIMENT PREDICTOR  (type 'quit' to exit)")
print(f"   🤖  Model in use : {best_name}")
print("=" * 56)

EMOJI_MAP = {"positive": "😊", "neutral": "😐", "negative": "😠"}

while True:
    user_input = input("\n  Enter a review: ").strip()

    if user_input.lower() in ["quit", "exit", "q"]:
        print("\n  👋  Exiting. Goodbye!\n")
        break

    if not user_input:
        print("  ⚠️   Please enter some text.")
        continue

    processed  = preprocess(user_input)
    input_vec  = vectorizer.transform([processed])
    pred_idx   = best_model.predict(input_vec)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    emoji      = EMOJI_MAP.get(pred_label.lower(), "🔍")

    # Show confidence if model supports predict_proba
    confidence_str = ""
    if hasattr(best_model, "predict_proba"):
        proba      = best_model.predict_proba(input_vec)[0]
        confidence = proba[pred_idx] * 100
        confidence_str = f"\n      Confidence  : {confidence:.1f}%"

    print(f"\n  {emoji}  Prediction  : {pred_label.upper()}{confidence_str}")
    print(f"      Model Used  : {best_name}")