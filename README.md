# Text Classification using Wikipedia

This project is part of the Natural Language Processing (NLP) course for the Artificial Intelligence Master's degree at the University of Verona. The goal is to classify texts into two categories: geographic and non-geographic, based on the content extracted from Wikipedia articles. The implementation utilizes natural language processing (NLP) techniques and machine learning algorithms.

## Project Structure

### 1. Problem Statement
The task is to attribute a class (geographic or non-geographic) to a given text input, leveraging NLP and machine learning methodologies.

### 2. Data Collection
The project retrieves text data from Wikipedia pages using the Wikipedia API. Annotated keywords are used to distinguish between geographic and non-geographic topics.

### 3. Preprocessing
Text preprocessing involves tokenization, stop word removal, and optional stemming or lemmatization to clean and normalize the text data.

### 4. Feature Extraction
Feature extraction is performed using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. Top nouns extracted from Wikipedia articles serve as features for classification.

### 5. Model Training
Two classification models are employed:
- Logistic Regression
- Naive Bayes

The models are trained on the extracted features and evaluated using metrics such as accuracy, precision, and recall.

### 6. Model Evaluation
The trained models are evaluated on a test dataset to assess their performance in classifying geographic and non-geographic texts.

### 7. Prediction
Given a text input, the models predict whether the content is geographic or non-geographic based on the learned patterns.

## Pipeline

1. **Data Collection**: Wikipedia API is utilized to fetch text data.
2. **Preprocessing**: Text data is cleaned and normalized through tokenization, stop word removal, and optional stemming or lemmatization.
3. **Feature Extraction**: Top nouns from Wikipedia articles are extracted and used as features for classification.
4. **Model Training**: Logistic Regression and Naive Bayes classifiers are trained on the extracted features.
5. **Model Evaluation**: The trained models are evaluated on a test dataset to measure their performance.
6. **Prediction**: Given a text input, the models predict its class (geographic or non-geographic).

## Technologies Used

- Python programming language
- Libraries:
  - NLTK (Natural Language Toolkit) for NLP tasks
  - Scikit-learn for machine learning algorithms
  - Wikipedia-API for accessing Wikipedia content


