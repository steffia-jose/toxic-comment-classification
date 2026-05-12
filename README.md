# Toxic Comment Classification — NLP & Deep Learning

## Overview
Benchmarked 5 deep learning models for toxic comment detection on 159,000 Wikipedia 
comments from the Kaggle Toxic Comment Classification Challenge.

## Models Compared
- Logistic Regression (baseline)
- Feedforward Neural Network (FNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Attention Mechanism

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.9386 | 0.7993 | 0.5288 | 0.6365 |
| FNN | 0.9480 | 0.8193 | 0.6274 | 0.7106 |
| RNN | 0.9255 | 0.6816 | 0.5008 | 0.5774 |
| LSTM | 0.9364 | 0.7927 | 0.5066 | 0.6182 |
| Attention | 0.9448 | 0.8220 | 0.5837 | 0.6826 |

**Best model: FNN (F1 = 0.71) | Attention model highest accuracy (94.5%)**

## Methodology
- Binarised 6 toxicity labels into Toxic (1) / Non-toxic (0)
- Trained Word2Vec embeddings (vector size=100, window=5) using Gensim
- Stratified 80/20 train-test split to preserve class balance
- Evaluated on Accuracy, Precision, Recall, and F1-score

## Tools & Libraries
Python, TensorFlow, Keras, Gensim, Scikit-learn, Pandas, NumPy, Matplotlib

## Dataset
Toxic Comment Classification Challenge — Kaggle  
~159,000 Wikipedia talk page comments labelled by human annotators
