================================================================
  🎯  SENTIMENT ANALYSIS PIPELINE
  Multi-Class Sentiment Classification using NLP + ML
================================================================

OVERVIEW
--------
This project classifies text reviews into three sentiment
categories: Positive, Neutral, and Negative.

It uses a full NLP preprocessing pipeline, TF-IDF vectorization,
hyperparameter-tuned ML models, and a soft-voting ensemble
to find the best performing model automatically.

----------------------------------------------------------------

PROJECT STRUCTURE
-----------------
sentiment_analysis/
│
├── dataset.csv                  ← Raw input data
│
├── config.py                    ← All settings & file paths
├── pipeline_1_preprocess.py     ← Text cleaning & lemmatization
├── pipeline_2_eda.py            ← EDA visualizations
├── pipeline_3_train.py          ← TF-IDF + model training
├── pipeline_4_predict.py        ← Interactive prediction loop
├── run_all.py                   ← Master runner (run this!)
│
└── README.txt                   ← This file

After running, these files are generated automatically:
├── processed_data.csv           ← Cleaned text output
├── tfidf_matrix.npz             ← TF-IDF sparse matrix
├── vectorizer.pkl               ← Saved TF-IDF vectorizer
├── labels.pkl                   ← Label encoder
├── models.pkl                   ← Best trained model
├── eda_plots.png                ← Class distribution + top words
└── confusion_matrix.png         ← Best model confusion matrix

----------------------------------------------------------------

HOW TO RUN
----------
Make sure all required packages are installed first:

    pip install pandas numpy scikit-learn matplotlib nltk tqdm scipy

Then run the full pipeline with ONE command:

    python run_all.py

Or run individual stages:

    python run_all.py 1          → Preprocessing only
    python run_all.py 1 2        → Preprocessing + EDA
    python run_all.py 3          → Training only*
    python run_all.py 4          → Prediction only*

    * Stage 3 requires Stage 1 to have been run first.
    * Stage 4 requires Stage 3 to have been run first.

Each pipeline file can also be run directly:

    python pipeline_1_preprocess.py
    python pipeline_2_eda.py
    python pipeline_3_train.py
    python pipeline_4_predict.py

----------------------------------------------------------------

PIPELINE STAGES EXPLAINED
--------------------------
Stage 1 — Preprocessing
  • Loads dataset.csv
  • Lowercases, removes HTML tags, URLs, punctuation
  • Removes stopwords (keeps negation words like "not", "never")
  • POS-tagged lemmatization for accurate word forms
  • Saves processed_data.csv and labels.pkl

Stage 2 — EDA
  • Reads processed_data.csv
  • Plots sentiment class distribution (bar chart)
  • Plots top-20 most frequent words per sentiment class
  • Saves eda_plots.png

Stage 3 — Training
  • Fits TF-IDF vectorizer (unigrams + bigrams, 40k features)
  • Trains 4 models with GridSearchCV (cv=3):
      - Logistic Regression
      - Naive Bayes
      - SVM (LinearSVC)
      - SGD Classifier
  • Builds a soft-voting Ensemble of all 4 models
  • Selects best model by F1-Weighted score
  • Saves models.pkl, vectorizer.pkl, confusion_matrix.png

Stage 4 — Prediction
  • Loads models.pkl (best model + vectorizer)
  • Runs an interactive loop: type any review, get prediction
  • Shows sentiment label + confidence score
  • Type 'quit' to exit

----------------------------------------------------------------

MODELS COMPARED
---------------
  • Logistic Regression   (lbfgs solver, balanced class weights)
  • Naive Bayes           (MultinomialNB)
  • SVM                   (LinearSVC, balanced class weights)
  • SGD Classifier        (modified_huber loss)
  • Voting Ensemble       (soft voting across all 4 models)

Best model is selected automatically by F1-Weighted score.

----------------------------------------------------------------

REQUIREMENTS
------------
  Python        >= 3.8
  pandas        >= 1.3
  numpy         >= 1.21
  scikit-learn  >= 1.0
  matplotlib    >= 3.4
  nltk          >= 3.6
  tqdm          >= 4.60
  scipy         >= 1.7

Install all at once:
  pip install pandas numpy scikit-learn matplotlib nltk tqdm scipy

----------------------------------------------------------------

DATASET FORMAT
--------------
The dataset.csv must contain these two columns:
  • cleaned_review       — the review text
  • sentiment_category   — label: Positive / Neutral / Negative

----------------------------------------------------------------
