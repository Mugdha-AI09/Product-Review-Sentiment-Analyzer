# ============================================================
# 💬  pipeline_4_predict.py
#     INPUT  : models.pkl  (saved by pipeline_3)
#     Runs the interactive prediction loop
# ============================================================

import pickle
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from config import MODELS_PKL

for pkg in ["stopwords", "wordnet", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))
negation_words = {"no", "not", "nor", "never", "neither", "without",
                  "nobody", "nothing", "nowhere", "hardly", "barely",
                  "doesn't", "isn't", "wasn't", "wouldn't", "couldn't"}
stop_words -= negation_words

EMOJI_MAP = {"positive": "😊", "neutral": "😐", "negative": "😠"}


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
    print("   Loading saved model ...", end=" ", flush=True)
    with open(MODELS_PKL, "rb") as f:
        saved = pickle.load(f)

    best_model = saved["best_model"]
    best_name  = saved["best_name"]
    vectorizer = saved["vectorizer"]
    le         = saved["label_encoder"]
    print("done ✅")

    print()
    print("=" * 56)
    print("   💬  SENTIMENT PREDICTOR  (type 'quit' to exit)")
    print(f"   🤖  Model in use : {best_name}")
    print("=" * 56)

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

        confidence_str = ""
        if hasattr(best_model, "predict_proba"):
            proba          = best_model.predict_proba(input_vec)[0]
            confidence     = proba[pred_idx] * 100
            confidence_str = f"\n      Confidence  : {confidence:.1f}%"

        print(f"\n  {emoji}  Prediction  : {pred_label.upper()}{confidence_str}")
        print(f"      Model Used  : {best_name}")


if __name__ == "__main__":
    run()
