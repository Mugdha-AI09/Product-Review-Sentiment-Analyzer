# ============================================================
# 🤖  pipeline_3_train.py
#     INPUT  : processed_data.csv  +  labels.pkl
#     OUTPUT : tfidf_matrix.npz  +  vectorizer.pkl
#              models.pkl  +  confusion_matrix.png
# ============================================================

import pickle
import warnings
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from config import (
    PROCESSED_CSV, LABELS_PKL, TFIDF_MATRIX, VECTORIZER_PKL,
    MODELS_PKL, CM_PLOT_PATH, TFIDF_CONFIG, TEST_SIZE, RANDOM_STATE
)

warnings.filterwarnings("ignore")


def run():
    print("=" * 56)
    print("   🤖  PIPELINE 3 : TRAINING")
    print("=" * 56)

    # ── Load data ──────────────────────────────────────────────
    df = pd.read_csv(PROCESSED_CSV)
    with open(LABELS_PKL, "rb") as f:
        label_data = pickle.load(f)
    le = label_data["label_encoder"]
    y  = df['sentiment']

    # ── Train / test split ────────────────────────────────────
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['processed_review'],
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # ── TF-IDF ────────────────────────────────────────────────
    print("\n   Fitting TF-IDF vectorizer ...")

    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Save vectorized dataset for reference
    X = vectorizer.transform(df['processed_review'])
    sp.save_npz(TFIDF_MATRIX, X)

    with open(VECTORIZER_PKL, "wb") as f:
     pickle.dump(vectorizer, f)
    print(f"   TF-IDF shape: {X.shape}  →  saved ✅")

    # ── Dataset summary ───────────────────────────────────────
    print()
    print("=" * 56)
    print(f"  Total samples  : {len(df):,}")
    print(f"  Shape of X     : {X.shape}")
    print("  Classes Mapping:")
    for label in le.classes_:
        idx = le.transform([label])[0]
        cnt = (df['sentiment_category'] == label).sum()
        print(f"    {label:>10}  →  {idx}   ({cnt:,} samples)")


    # ── Hyperparameter grids ──────────────────────────────────
    param_grids = {
        "Logistic Regression": {
            "estimator" : LogisticRegression(max_iter=1000, solver="lbfgs",
                                             class_weight="balanced"),
            "param_grid": {"C": [0.1, 1, 5, 10]},
        },
        "Naive Bayes": {
            "estimator" : MultinomialNB(),
            "param_grid": {"alpha": [0.02, 0.05, 0.1, 0.2, 0.5]},
        },
        "SVM (LinearSVC)": {
            "estimator" : LinearSVC(max_iter=3000, dual=True,
                                    class_weight="balanced"),
            "param_grid": {"C": [0.5, 1, 2, 5, 10, 20]},
        },
        "SGD Classifier": {
            "estimator" : SGDClassifier(loss="modified_huber", max_iter=1000,
                                        class_weight="balanced",
                                        random_state=RANDOM_STATE),
            "param_grid": {
                "alpha"  : [1e-5, 1e-4, 1e-3, 1e-2],
                "penalty": ["l2", "elasticnet"],
            },
        },
    }

    # ── Train individual models ────────────────────────────────
    print()
    print("=" * 60)
    print("   🔧  HYPERPARAMETER TUNING  (GridSearchCV, cv=3)")
    print("=" * 60)

    results   = {}
    best_ests = {}

    for name, cfg in param_grids.items():
        print(f"\n  Tuning {name} ...", end=" ", flush=True)
        gs_cv = GridSearchCV(cfg["estimator"], cfg["param_grid"],
                             cv=3, scoring="f1_weighted", n_jobs=-1)
        gs_cv.fit(X_train, y_train)
        best_est        = gs_cv.best_estimator_
        best_ests[name] = best_est
        y_pred          = best_est.predict(X_test)
        results[name]   = _metrics(best_est, gs_cv.best_params_, y_test, y_pred, le)
        print(f"done ✅  best params: {gs_cv.best_params_}")

    # ── Voting Ensemble ───────────────────────────────────────
    print(f"\n  Building Voting Ensemble ...", end=" ", flush=True)
    best_svm_C     = best_ests["SVM (LinearSVC)"].C
    calibrated_svm = CalibratedClassifierCV(
        LinearSVC(C=best_svm_C, max_iter=3000, dual=True,
                  class_weight="balanced"), cv=3)

    ensemble = VotingClassifier(
        estimators=[
            ("lr",  best_ests["Logistic Regression"]),
            ("nb",  best_ests["Naive Bayes"]),
            ("svm", calibrated_svm),
            ("sgd", best_ests["SGD Classifier"]),
        ],
        voting="soft", n_jobs=1,
    )
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    results["Voting Ensemble"] = _metrics(
        ensemble, {"voting": "soft", "models": "LR+NB+SVM+SGD"},
        y_test, y_pred_ens, le)
    print("done ✅")

    # ── Metrics table ─────────────────────────────────────────
    _print_metrics(results)

    # ── Best model ────────────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k]["f1_weighted"])
    best       = results[best_name]
    best_model = best["model"]
    _print_best(best_name, best)

    # ── Confusion matrix ──────────────────────────────────────
    _save_confusion_matrix(best_name, best["y_pred"], y_test, le)

    # ── Save everything ───────────────────────────────────────
    with open(MODELS_PKL, "wb") as f:
        pickle.dump({
            "best_model"   : best_model,
            "best_name"    : best_name,
            "results"      : results,
            "vectorizer"   : vectorizer,
            "label_encoder": le,
        }, f)
    print(f"\n   Saved → models.pkl ✅\n")


# ── Helpers ───────────────────────────────────────────────────

def _metrics(model, best_params, y_test, y_pred, le):
    return {
        "model"       : model,
        "best_params" : best_params,
        "accuracy"    : accuracy_score(y_test, y_pred) * 100,
        "f1_weighted" : f1_score(y_test, y_pred, average="weighted") * 100,
        "f1_macro"    : f1_score(y_test, y_pred, average="macro")    * 100,
        "precision"   : precision_score(y_test, y_pred, average="weighted") * 100,
        "recall"      : recall_score(y_test, y_pred, average="weighted")    * 100,
        "y_pred"      : y_pred,
        "report"      : classification_report(y_test, y_pred,
                                               target_names=le.classes_),
    }


def _print_metrics(results):
    print()
    print("=" * 72)
    print("                    📈  MODEL METRICS COMPARISON")
    print("=" * 72)
    COL_W  = 20
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
    for name, res in results.items():
        bar_len = int(res["accuracy"] / 2)
        bar     = "█" * bar_len + "░" * (50 - bar_len)
        print(f"  {name}\n  {res['accuracy']:.2f}%  [{bar}]\n")


def _print_best(best_name, best):
    print("=" * 72)
    print("                        🏆  BEST MODEL")
    print("=" * 72)
    print(f"  ✅  {best_name}")
    print(f"      Best params  : {best['best_params']}")
    print(f"      Accuracy     : {best['accuracy']:.2f}%")
    print(f"      F1-Weighted  : {best['f1_weighted']:.2f}%")
    print(f"      F1-Macro     : {best['f1_macro']:.2f}%")
    print(f"      Precision    : {best['precision']:.2f}%")
    print(f"      Recall       : {best['recall']:.2f}%")
    print()
    print("  Classification Report:")
    print(best["report"])
    print("=" * 72)


def _save_confusion_matrix(best_name, y_pred, y_test, le):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    cm  = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    cmd.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}",
                 color="white", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    plt.setp(ax.get_xticklabels(), color="white")
    plt.setp(ax.get_yticklabels(), color="white")
    plt.tight_layout()
    plt.savefig(CM_PLOT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   Saved → confusion_matrix.png ✅")


if __name__ == "__main__":
    run()
