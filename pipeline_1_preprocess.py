# ============================================================
# 🧹  pipeline_1_preprocess.py
#     INPUT  : dataset.csv
#     OUTPUT : processed_data.csv  +  labels.pkl
# ============================================================

import re
import pickle
import warnings
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from config import DATA_PATH, PROCESSED_CSV, LABELS_PKL

warnings.filterwarnings("ignore")

# ── One-time NLTK downloads ───────────────────────────────────
for pkg in ["stopwords", "wordnet", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# ── Setup ─────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

negation_words = {"no", "not", "nor", "never", "neither", "without",
                  "nobody", "nothing", "nowhere", "hardly", "barely",
                  "doesn't", "isn't", "wasn't", "wouldn't", "couldn't"}
stop_words -= negation_words


def _get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"): return wordnet.ADJ
    if treebank_tag.startswith("V"): return wordnet.VERB
    if treebank_tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN


def preprocess(text: str) -> str:
    text   = str(text).lower()
    text   = re.sub(r"<.*?>",          " ", text)
    text   = re.sub(r"http\S+|www\S+", " ", text)
    text   = re.sub(r"[^a-z\s]",       " ", text)
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 2]
    tagged = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(tok, _get_wordnet_pos(pos)) for tok, pos in tagged]
    return " ".join(tokens)


def run():
    print("=" * 56)
    print("   🧹  PIPELINE 1 : PREPROCESSING")
    print("=" * 56)

    df = pd.read_csv(DATA_PATH)
    df = df[['cleaned_review', 'sentiment_category']].dropna()
    print(f"   Loaded {len(df):,} rows from dataset.csv")

    tqdm.pandas(desc="   Processing rows")
    df['processed_review'] = df['cleaned_review'].progress_apply(preprocess)

    # Encode labels
    le = LabelEncoder()
    df['sentiment'] = le.fit_transform(df['sentiment_category'])

    # Save processed CSV
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"   Saved → processed_data.csv ✅")

    # Save label encoder
    with open(LABELS_PKL, "wb") as f:
        pickle.dump({"label_encoder": le, "y": df['sentiment']}, f)
    print(f"   Saved → labels.pkl ✅\n")


if __name__ == "__main__":
    run()
