# Multi-class Classification

## Introduction
While Binary classification produced great results, I still wanted to try out some popular multi-class classification algorithms and techniques. However, for this section, following the initial bad results, I decided to turn this into a petri-dish for testing different algorithms. Hence, this section will just be a summary of results from using all the different algorithms, with a sprinkle of observations here and there. 

Overall, I broke the task into three sections based on the input that I was giving to the algorithms:
- **TF-IDF vectors**
- **Word2Vec Embeddings**
- **BERT Embeddings**

For all of the above three inputs, I used three different classifiers - **OneVsRest with Logistic Regression**, **OneVsOne with Support Vector Classification (SVC)** and **XGBoost**

---

## Part A: Algorithms with TF-IDF Vectors

### Results
For this section, I removed class-imbalance by choosing an equal number of data points from all classes. Based on past experience with Binary Classification, I chose to create TF-IDF vectors where the features could be unigrams or bigrams, and the maximum features were set to 7000. Here are the results for all three algorithms:

### OneVsRest with Logistic Regression:
- **ROC-AUC Score:** 0.8055264277807354

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 1     | 0.52      | 0.57   | 0.55     | 2198    |
| 2     | 0.37      | 0.35   | 0.36     | 2198    |
| 3     | 0.39      | 0.37   | 0.38     | 2198    |
| 4     | 0.46      | 0.44   | 0.45     | 2198    |
| 5     | 0.55      | 0.59   | 0.57     | 2198    |

**Overall Metrics:**
- **Accuracy:** 0.46
- **Macro Avg:** Precision: 0.46, Recall: 0.46, F1-score: 0.46
- **Weighted Avg:** Precision: 0.46, Recall: 0.46, F1-score: 0.46

---

### OneVsOne with Support Vector Classification:
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 1     | 0.45      | 0.55   | 0.49     | 2198    |
| 2     | 0.34      | 0.27   | 0.30     | 2198    |
| 3     | 0.33      | 0.30   | 0.31     | 2198    |
| 4     | 0.40      | 0.35   | 0.37     | 2198    |
| 5     | 0.49      | 0.60   | 0.54     | 2198    |

**Overall Metrics:**
- **Accuracy:** 0.41
- **Macro Avg:** Precision: 0.40, Recall: 0.41, F1-score: 0.41
- **Weighted Avg:** Precision: 0.40, Recall: 0.41, F1-score: 0.41

---

### XGBoost:
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 1     | 0.52      | 0.58   | 0.55     | 2198    |
| 2     | 0.38      | 0.37   | 0.37     | 2198    |
| 3     | 0.39      | 0.32   | 0.36     | 2198    |
| 4     | 0.45      | 0.41   | 0.43     | 2198    |
| 5     | 0.52      | 0.61   | 0.56     | 2198    |

**Overall Metrics:**
- **Accuracy:** 0.46
- **Macro Avg:** Precision: 0.45, Recall: 0.46, F1-score: 0.45
- **Weighted Avg:** Precision: 0.45, Recall: 0.46, F1-score: 0.45

---

### General Observations

All three algorithms achieved similar results. **OvR and XGBoost** had slightly better results than OvO - however, the patterns in class-wise predictions are similar. The F1 score for Ratings 1 and 5 were better than the middle ratings across all three algorithms. The same can be observed for precision and recall. Specifically, our algorithms struggle with distinguishing between Ratings 1 and 2 in our dataset. 


This highlights the biggest challenge in rating classification — **personal bias.** Two people who wrote similar reviews could give completely different scores based on personal preference. This is even more evident in the lower ratings where the choice between a 2 and 3 truly boils down to personal preference. 

---

## Part B: Algorithms with Word2Vec Embeddings

### Results
For this section, I removed class-imbalances as well. Similar to its binary version, I used Word2Vec averaging to create a single multi-dimensional vector representation for each of my reviews.

### OneVsRest with Logistic Regression:
- **ROC-AUC Score:** 0.8018057920965458

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 1     | 0.51      | 0.62   | 0.56     | 2198    |
| 2     | 0.38      | 0.33   | 0.35     | 2198    |
| 3     | 0.39      | 0.35   | 0.37     | 2198    |
| 4     | 0.45      | 0.44   | 0.45     | 2198    |
| 5     | 0.57      | 0.60   | 0.59     | 2198    |

**Overall Metrics:**
- **Accuracy:** 0.47
- **Macro Avg:** Precision: 0.46, Recall: 0.47, F1-score: 0.46
- **Weighted Avg:** Precision: 0.46, Recall: 0.47, F1-score: 0.46

---

# OneVsOne with Support Vector Classifier

| Rating | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1      | 0.55      | 0.46   | 0.50     | 2198    |
| 2      | 0.37      | 0.36   | 0.36     | 2198    |
| 3      | 0.38      | 0.46   | 0.42     | 2198    |
| 4      | 0.42      | 0.41   | 0.42     | 2198    |
| 5      | 0.54      | 0.56   | 0.55     | 2198    |

**Overall Metrics:**
- **Accuracy:** 0.45
- **Macro Avg:** Precision: 0.45, Recall: 0.45, F1-score: 0.45
- **Weighted Avg:** Precision: 0.45, Recall: 0.45, F1-score: 0.45

---

# XGBoost

| Rating | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1      | 0.52      | 0.58   | 0.55     | 2198    |
| 2      | 0.39      | 0.37   | 0.38     | 2198    |
| 3      | 0.36      | 0.33   | 0.35     | 2198    |
| 4      | 0.44      | 0.44   | 0.44     | 2198    |
| 5      | 0.56      | 0.55   | 0.56     | 2198    |

**Overall Metrics:**
- **Accuracy:** 0.46
- **Macro Avg:** Precision: 0.45, Recall: 0.46, F1-score: 0.45
- **Weighted Avg:** Precision: 0.45, Recall: 0.46, F1-score: 0.45

---

# General Observations

With **Word2Vec Embeddings**, I achieved near-identical average F1-scores for all three of our algorithms. However, I observed that OvO’s performance was more evenly distributed amongst Ratings 3 and 4 compared to the other algorithms. This could be due to the nature of the algorithm, which specifically trains the model to distinguish between Ratings 3 and 4.

---

# Part C: Algorithms with BERT Embeddings

**BERT** is a super-popular pre-trained model made by Google that stands for **Bidirectional Encoder Representations from Transformers**. I used this pre-trained model to first tune it to my dataset and then turn it into multi-dimensional vectors (similar to Word2Vec) that represent my reviews. Since this was my first time using BERT Embeddings, I tested the model performance on three dataset versions:

1. **Original dataset**
2. **Class-imbalance removed by undersampling**
3. **Rating 3 removed altogether**


## Original Dataset:

### One-vs-Rest Logistic Regression Evaluation

| Rating | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1      | 0.11      | 0.86   | 0.19     | 2250    |
| 2      | 0.38      | 0.06   | 0.10     | 3538    |
| 3      | 0.45      | 0.07   | 0.12     | 6296    |
| 4      | 0.63      | 0.68   | 0.65     | 28688   |
| 5      | 0.65      | 0.30   | 0.41     | 19005   |

**Overall Metrics:**
- **Accuracy:** 0.47
- **Macro Avg:** Precision: 0.44, Recall: 0.39, F1-score: 0.30
- **Weighted Avg:** Precision: 0.58, Recall: 0.47, F1-score: 0.47

---

### XGBoost Evaluation

| Rating | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1      | 0.42      | 0.27   | 0.33     | 2249    |
| 2      | 0.33      | 0.23   | 0.27     | 3538    |
| 3      | 0.37      | 0.28   | 0.32     | 6296    |
| 4      | 0.59      | 0.77   | 0.67     | 28688   |
| 5      | 0.58      | 0.41   | 0.48     | 19005   |

**Overall Metrics:**
- **Accuracy:** 0.55
- **Macro Avg:** Precision: 0.38, Recall: 0.33, F1-score: 0.35
- **Weighted Avg:** Precision: 0.54, Recall: 0.55, F1-score: 0.54

---

## Class-Imbalance Removed by Undersampling

### One-vs-Rest Logistic Regression Evaluation

| Rating | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1      | 0.28      | 0.92   | 0.42     | 2243    |
| 2      | 0.48      | 0.10   | 0.16     | 2190    |
| 3      | 0.49      | 0.11   | 0.18     | 2234    |
| 4      | 0.53      | 0.29   | 0.38     | 2156    |
| 5      | 0.63      | 0.41   | 0.50     | 2167    |

**Overall Metrics:**
- **Accuracy:** 0.37
- **Macro Avg:** Precision: 0.48, Recall: 0.36, F1-score: 0.33
- **Weighted Avg:** Precision: 0.48, Recall: 0.37, F1-score: 0.33

---

### XGBoost Evaluation

| Rating | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1      | 0.51      | 0.54   | 0.53     | 2243    |
| 2      | 0.37      | 0.37   | 0.37     | 2190    |
| 3      | 0.38      | 0.34   | 0.36     | 2234    |
| 4      | 0.44      | 0.43   | 0.44     | 2156    |
| 5      | 0.54      | 0.57   | 0.56     | 2167    |

**Overall Metrics:**
- **Accuracy:** 0.45
- **Macro Avg:** Precision: 0.45, Recall: 0.45, F1-score: 0.45
- **Weighted Avg:** Precision: 0.45, Recall: 0.45, F1-score: 0.45

---

## Rating 3 Removed:

### One-vs-Rest Logistic Regression Evaluation

| Rating | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1      | 0.41      | 0.82   | 0.54     | 2228    |
| 2      | 0.59      | 0.30   | 0.40     | 2159    |
| 4      | 0.58      | 0.45   | 0.51     | 2188    |
| 5      | 0.66      | 0.45   | 0.53     | 2217    |

**Overall Metrics:**
- **Accuracy:** 0.51
- **Macro Avg:** Precision: 0.56, Recall: 0.50, F1-score: 0.50
- **Weighted Avg:** Precision: 0.56, Recall: 0.51, F1-score: 0.50

---

### XGBoost Evaluation

| Rating | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1      | 0.57      | 0.58   | 0.57     | 2228    |
| 2      | 0.51      | 0.51   | 0.51     | 2159    |
| 4      | 0.52      | 0.50   | 0.51     | 2188    |
| 5      | 0.58      | 0.59   | 0.59     | 2217    |

**Overall Metrics:**
- **Accuracy:** 0.55
- **Macro Avg:** Precision: 0.54, Recall: 0.54, F1-score: 0.54
- **Weighted Avg:** Precision: 0.54, Recall: 0.55, F1-score: 0.54

---

# General Observations

Across all dataset versions, **XGBoost** consistently performed better. When considering the dataset without addressing class imbalance, I achieved a much higher F1 score for the dominant class, but lost significant accuracy for others. The weighted average however, looks much better. After balancing the classes, I achieved more equitable performance, though **OvR Logistic Regression** suffered. Removing the Rating 3 class entirely produced the best results (55% accuracy). 

However, removing a class entirely is not ideal in a multi-class classification problem. The most meaningful improvement would be achieving comparable or better results without ignoring any classes.

---

# Possible Improvements:

- I would like to try and use LSTMs and GRUs on this dataset for multi-class classification. 
- I would also like to consider using different sets of algorithms for different sets of rating-groups. So perhaps one algorithm to distinguish between positive and negative, and then another to distinguish between 1 & 2 and 3 & 4.

