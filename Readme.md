# Project Overview: NLP Exploration

For this project, I had just one goal in mind — explore different algorithms, tools, and techniques in the world of Natural Language Processing. Hence, at times this project would look chaotic or even a bit redundant. However, at the core of it all was sheer will to just ‘explore’ and ‘try’ new things.

For the sake of this exploration, I used a Car Reviews dataset that has two main columns — `Review` and `Rating`. This makes it a perfect dataset to try out Supervised Machine Learning algorithms for Classification. The `Ratings` are scored on a scale of 1 to 5. This allows for multi-class classification and binary classification — if we split the data into ‘negative’ and ‘positive’. But we have 5 different classes. So how do we split them into two classes? That’s something to be explored!

This repository explores sentiment analysis of car reviews through comprehensive data analysis and classification tasks. It is structured into three key sections:
- **Exploratory Data Analysis (EDA)**
- **Binary Classification**
- **Multi-class Classification**

Each folder in the repository corresponds to a specific phase of the project, with each one having its own `README` file detailing methodologies, algorithms, and tools used.

**Scroll down to the end to see a quick summary of the Libraries used!**

---

## 1. Exploratory Data Analysis (EDA)

**Goal:** To gain insights into the Car Reviews Dataset, which contains customer reviews rated from 1 to 5.

[Click here for EDA Report and Code](https://github.com/evlasnoraa/NLP-Exploration/tree/main/EDA)

### Dataset Characteristics:
- Understanding our two columns: `Rating` (1-5) and `Review` (text data).
- Visualizing imbalanced class distribution, with `Ratings` 4 and 5 dominating.

### Key Insights:
- Word frequency analysis highlights terms like *drive*, *comfort*, and *engine issue*.
- Analysis of review length reveals changes in review length with respect to `Rating`.
- Polarity scores derived using `TextBlob` confirm alignment with ratings but highlight challenges in distinguishing mid-range sentiments.

### Tools and Techniques:
- **Libraries:** `Python` (`Pandas`, `Matplotlib`, `Nltk`, `Scikit-learn`, `Seaborn`, `WordCloud`, `TextBlob`).
- **Preprocessing:** Removal of stopwords, stemming, tokenization, and bi-gram extraction for textual analysis.
- **Methodology:** Use of the above-mentioned libraries to develop summary statistics, box plots, bar graphs, and word clouds.

---

## 2. Binary Classification

**Goal:** To classify reviews as either positive or negative, leveraging the imbalanced dataset effectively.

[Click here for Binary Classification Report and Code](https://github.com/evlasnoraa/NLP-Exploration/tree/main/Binary%20Classification)

### Methodology:
1. Exploring different ways of splitting data using four different approaches with `Multinomial Naive Bayes` as a baseline.
2. Using the best approach on the rest of the algorithms.

### Algorithms Used:
- **Baseline:** `Multinomial Naive Bayes` with `TF-IDF` vectorization to judge and shortlist one of the four approaches.
- **Linear Models:** `Logistic Regression` and `Linear SVMs` with `TF-IDF` and `Word2Vec` vectorization techniques.
- **Tree-based Models:** `Random Forest`, `XGBoost`, and ensemble methods developed from a combination of all the above algorithms.

### Tools & Techniques:
- **Preprocessing:** Used `Scikit-learn`’s `TfidfVectorizer` to create `TF-IDF` vectors, and the `Gensim` library for `Word2Vec` embeddings.
- **Model Training:** Used `Scikit-learn`’s `MultinomialNB`, `LogisticRegression`, `LinearSVC`, `RandomForestClassifier`, `StackingClassifier`, and `XGBClassifier` to implement the above algorithms.
- **Model Tuning:** `GridSearchCV`, Bayesian optimization (`Optuna`). Used `Scikit-learn`’s `pipeline` to streamline the vectorization and model training process into one streamlined process.
- **Model Evaluation:** Used several different metrics like `F1 score` (`f1_score`), `Accuracy score` (`accuracy_score`), and `ROC-AUC Score` (`roc_auc_score`). Also used `Confusion matrix` (`confusion_matrix`) for more detailed class-wise results.

---

## 3. Multi-class Classification

**Goal:** To classify reviews into five distinct ratings, addressing the complexities of class overlap and subjective sentiment.

[Click here for Mutli-class Classification Report and Code](https://github.com/evlasnoraa/NLP-Exploration/tree/main/Multi-class%20Classification)

### Challenges:
- Overlapping sentiments in adjacent ratings (e.g., 2 and 3).
- Imbalanced dataset requiring undersampling for fair evaluation.

### Approaches Tested:
- **Input Types:** `TF-IDF`, `Word2Vec` embeddings, `BERT` embeddings.
- **Models:** `Logistic Regression` (One-vs-Rest), `SVMs` (One-vs-One), and `XGBoost`.

### Tools & Techniques:
Since this part of the project was built upon Binary Classification, I used all the libraries from before and the following in addition:
- **Preprocessing:** Used the `Transformers` library to get the `BERT` model (`BertTokenizer`, `BertModel`, etc.).
- **Model Training:** Used `Scikit-learn`’s `OneVsOneClassifier` and `OneVsRestClassifier`, which work really well with `Scikit-learn`’s other models mentioned above.
- **Model Tuning:** No model tuning done for this section!
- **Model Evaluation:** Since this was multi-class classification, I used `classification_report` that gives overall and class-wise precision, recall, and F1 scores. Very helpful for a thorough analysis.

---

## Libraries Summary:
- **Data Manipulation and Analysis:**
  - `Pandas`: For handling datasets, preprocessing, and generating summary statistics.
  - `Numpy`: For numerical operations.
- **Visualization:**
  - `Matplotlib` and `Seaborn`: For creating box plots, bar graphs, and detailed visualizations.
  - `WordCloud`: For generating word clouds to analyze word frequency.
- **Natural Language Processing:**
  - `NLTK`: For stopword removal, stemming, tokenization, and n-gram extraction.
  - `TextBlob`: For sentiment polarity analysis.
  - `Gensim`: For generating `Word2Vec` embeddings.
  - `Transformers`: For working with `BERT` embeddings (`BertTokenizer`, `BertModel`, etc.).
- **Machine Learning:**
  - `Scikit-learn`: For vectorization (`TfidfVectorizer`), implementing machine learning models (`MultinomialNB`, `LogisticRegression`, `LinearSVC`, `RandomForestClassifier`, `OneVsOneClassifier`, and `OneVsRestClassifier`), model pipelines, and evaluation metrics (`f1_score`, `accuracy_score`, `roc_auc_score`, and `confusion_matrix`).
  - `XGBoost`: For tree-based classification.
  - `Optuna`: For Bayesian optimization during model tuning.
 
  ---
