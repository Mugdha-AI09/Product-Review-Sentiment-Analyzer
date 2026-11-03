# 🛍️ Product Review Sentiment Analyzer

An NLP and Machine Learning-based application that analyzes customer product reviews and classifies them into three sentiment categories: **Positive, Negative, and Neutral**.

The project implements an end-to-end sentiment analysis pipeline, including text preprocessing, exploratory data analysis, TF-IDF feature extraction, model training, hyperparameter tuning, evaluation, and real-time sentiment prediction through an interactive Tkinter GUI.

## ✨ Features

* Text cleaning and preprocessing
* Stopword removal and lemmatization
* Exploratory Data Analysis (EDA)
* TF-IDF feature extraction
* Multiple Machine Learning classifiers
* Hyperparameter optimization using GridSearchCV
* Soft Voting Ensemble
* Model evaluation using classification metrics
* Interactive Tkinter GUI
* Real-time sentiment prediction

## 🧠 Machine Learning Models

The project implements and compares:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine
4. SGD Classifier
5. Soft Voting Ensemble

The best-performing model is selected based on weighted F1-score.

## 📊 Model Performance

| Metric            | SGD Classifier |
| ----------------- | -------------: |
| Accuracy          |         91.77% |
| Weighted F1-score |         91.02% |
| Macro F1-score    |         56.95% |
| Precision         |         90.73% |
| Recall            |         91.77% |

**Note:** The dataset has an imbalanced class distribution, particularly for Neutral sentiment. Consequently, minority-class performance is lower than overall accuracy suggests.

## 📈 Visualizations

The project generates:

* Sentiment distribution and word-frequency visualizations.
* Confusion matrix for model evaluation.

## 🛠️ Technologies Used

* Python
* Machine Learning
* Natural Language Processing (NLP)
* Scikit-learn
* NLTK
* Pandas
* NumPy
* Matplotlib
* SciPy
* Tkinter

## ⚙️ Installation and Setup

Clone the repository:

```bash
git clone https://github.com/Mugdha-AI09/Product-Review-Sentiment-Analyzer.git
```

Navigate to the project folder:

```bash
cd Product-Review-Sentiment-Analyzer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the graphical application:

```bash
python gui.py
```

To execute the complete ML pipeline:

```bash
python run_all.py
```

## 📂 Project Structure

```text
Product-Review-Sentiment-Analyzer/
│
├── experiments/
│   └── sentiment_analysis_v2.py
│
├── config.py
├── gui.py
├── dataset.csv
├── processed_data.csv
├── pipeline_1_preprocess.py
├── pipeline_2_eda.py
├── pipeline_3_train.py
├── pipeline_4_predict.py
├── models.pkl
├── labels.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```
## Experimental Implementation

The repository also includes an experimental implementation of the sentiment analysis project in the `experiments/` directory.

### `sentiment_analysis_v2.py`

This standalone implementation explores an alternative approach to product review sentiment classification.

**Features:**

* Enhanced text preprocessing using POS-tagged lemmatization.
* Negation word preservation during stopword removal.
* TF-IDF feature extraction with unigram and bigram support.
* Multiple machine learning classifiers:

  * Logistic Regression
  * Multinomial Naive Bayes
  * Linear Support Vector Machine
  * SGD Classifier
* Hyperparameter tuning using GridSearchCV.
* Soft Voting Ensemble.
* Model evaluation using accuracy, precision, recall and F1-score.
* Confusion matrix visualisation.
* Interactive terminal-based sentiment prediction.

**Note:** This is an experimental standalone implementation and is separate from the main modular application. The primary project workflow and GUI are maintained independently.

## 🚀 Future Improvements

* Improve minority-class sentiment detection.
* Experiment with advanced NLP and transformer-based models.
* Integrate larger and more diverse datasets.
* Develop a web-based interface.
* Deploy the application for online access.

## 👩‍💻 Author

**Mugdha Pawar**

Artificial Intelligence and Machine Learning

[GitHub Profile](https://github.com/Mugdha-AI09)
