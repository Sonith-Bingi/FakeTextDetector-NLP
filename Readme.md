# Fake or Real: The Impostor Hunt

**Author:** Sonith Bingi  
**Competition:** [Kaggle - Fake or Real: The Impostor Hunt](https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt)

---

## Overview

This project addresses the challenge of identifying whether a given text sample is genuine or synthetically generated. The notebook builds a classification pipeline that combines feature engineering, natural language preprocessing, and machine learning models to distinguish between real and fake text with high accuracy.

The focus is on balancing interpretability with performance through traditional ML models, rather than transformer-based architectures.

---

## Approach

### Data Understanding
The dataset includes labeled examples of text classified as either *real* or *fake*. Each row contains a unique text sample and its corresponding label.

### Preprocessing
- Tokenization, lowercasing, and stopword removal  
- Lemmatization and punctuation cleaning  
- TF-IDF vectorization of both word and character n-grams  

### Modeling
Multiple classical machine learning models were trained and compared:
- Support Vector Machine (SVM)  
- Logistic Regression  
- Stochastic Gradient Descent (SGD) Classifier  

An ensemble of these models using majority voting achieved the highest validation performance.

### Evaluation Metrics
- Accuracy  
- F1 Score  
- Confusion Matrix for error analysis
