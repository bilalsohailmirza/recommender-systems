# Recommender Systems Project Report: Comparison of Collaborative Filtering and Neural Networks

This repository contains the project report and code for a comparative study of collaborative filtering and neural network-based recommender systems. The project utilizes the MovieLens dataset to evaluate the performance of both approaches in tailoring content to user preferences.

## Table of Contents
1.  [Introduction](#introduction)
2.  [Methodology](#methodology)
    * [Dataset](#dataset)
    * [Collaborative Filtering System](#collaborative-filtering-system)
    * [Neural Network-Based Recommender System](#neural-network-based-recommender-system)
    * [Comparison Metrics](#comparison-metrics)
3.  [Environment Setup](#environment-setup)
4.  [Results](#results)
5.  [Discussion](#discussion)
6.  [Conclusion](#conclusion)
7.  [References](#references)

## Introduction

Recommender systems play a pivotal role in various domains, such as e-commerce and streaming services, by tailoring content to users' preferences. This project aims to compare two fundamental approaches—collaborative filtering and neural network-based recommendations—based on their performance and applicability.

## Methodology

### Dataset

The project uses the MovieLens dataset, which contains:
* 100,000 ratings from 943 users on 1,682 movies.
* User demographic data (age, gender, occupation).
* Movie metadata (genres).

Key preprocessing steps involved merging user ratings with movie genres and splitting the data into training and testing sets.

### Collaborative Filtering System

The collaborative filtering system leverages user-item interactions.
* **Interaction Matrix:** User ratings are converted into an interaction matrix.
* **Similarity Matrix:** A similarity matrix between users is calculated based on their ratings. Missing ratings (NaN) are replaced with 0 to ensure consistency. Cosine similarity computes how similar each pair of users is, resulting in a matrix with similarity scores.
* **Predictions:** This function predicts missing ratings in the interaction matrix. It skips items already rated by the user. For unfilled entries, it calculates a weighted average of ratings given by similar users. The weights are the similarity scores.
* **Evaluation:** The Mean Squared Error (MSE) is calculated between the test interaction matrix and predicted interaction matrix, ignoring missing values in the test data.

**Strengths:**
* Exploits user-item interactions.
* Effective for dense datasets.

**Weaknesses:**
* Struggles with sparse datasets.
* Cold start problem for new users and items.

### Neural Network-Based Recommender System

The neural network-based recommender system models user-item interactions using deep learning techniques to model user preferences and is built with PyTorch.
* **Data Preparation:** The data preparation process for the Neural Network involves converting the feature matrices (X_train and X_test) and target labels (Y_train and Y_test) into PyTorch tensors. This step prepares the data for input into the neural network, which will automatically learn relationships between features during training.
* **Predictions:** In the Neural Network, predictions are made during the forward pass through the model. The input features are passed through several layers (including normalization, linear transformations, activations, and dropout), and the final layer (self.classifier) generates a prediction. These predictions are raw logits, which can be converted to probabilities using a sigmoid function.
* **Evaluation:** In the evaluation step, the Neural Network computes predictions for the test set. It uses a sigmoid function to convert raw logits into probabilities and thresholds them at 0.5 for binary classification. The accuracy is calculated as the percentage of correct predictions (predicted matches labels) over the total test samples.

**Strengths:**
* Can learn complex patterns in data.
* Handles non-linear relationships effectively.

**Weaknesses:**
* Requires more computational resources.
* Can overfit on small datasets.

### Comparison Metrics

The comparison is based on accuracy, scalability, and computational efficiency.
* **RMSE:** Measures prediction accuracy.
* **Diversity:** Evaluates the variety of recommendations.
* **Scalability:** Assessed based on runtime and memory usage.

## Environment Setup

The project uses Python and PyTorch. The specific libraries and versions can be found in the project's GitHub repository.

## Results

The following table summarizes the key results:

| Metric | Collaborative Filtering | Neural Network Filtering |
| :----- | :---------------------- | :----------------------- |
| Accuracy | 82.5238 | 83.72 |
| MSE | 11.11 | |

**Computational Efficiency:**
Collaborative filtering is more computationally efficient for small to medium datasets but struggles to scale with large, sparse data. In contrast, neural network-based recommenders are more resource-intensive and usually require accelerated computing through GPUs.

**Scalability:**
Neural network-based recommenders are more resource-intensive but offer better scalability through GPU acceleration and parallel processing, making them suitable for large-scale, complex systems.

## Discussion

Both systems have unique strengths:
* Collaborative filtering excels in leveraging user-item interactions effectively.
* Neural networks shine in capturing complex patterns but are computationally intensive.
Hybrid systems could combine the strengths of both for enhanced recommendations.

## Conclusion

This project demonstrates the trade-offs between collaborative filtering and neural network-based filtering. Each approach suits specific use cases, and their integration holds the potential for superior performance.

## References

* GroupLens Research: MovieLens Dataset
* Scikit-learn Documentation
* PyTorch Documentation

The code for this project is available on GitHub: [https://github.com/bilalsohailmirza/recommender-systems](https://github.com/bilalsohailmirza/recommender-systems)
