## A simple song recommender and analyzer based on audio features

# Overview

This project is a song recommendation system that suggests songs based on audio features extracted from music tracks. Using different techniques, it analyzes song characteristics and finds similar tracks based on feature similarity, clustering, and ranking methods. The system can be used to recommend songs with a similar style, genre, or mood to a given query song. Another option is to upload a song and analyze its features using an external API to call a pretrained model.

# How it works

The recommendation engine relies on audio feature analysis using different techniques, including:
- **Cosine Similarity**: Finds songs that are most similar to the query based on feature vectors.
- **Exponential Weighting**: Emphasizes important audio features to improve recommendations.
- **Radial Basis Function Similarity**: Uses non-linear kernel methods to enhance recommendations.
- **KMeans Clustering**: Groups songs into clusters and recommends songs from the closest cluster.
- **Annoy Index (Approximate Nearest Neighbors)**: Efficiently finds the closest matches in a large dataset.

# How to load the app

1. Open a terminal.
2. Go to the root directory of the app (path/to/dir).
3. Start a local server.
4. Run `python app.py`.
5. Open a browser.
6. Visit http://127.0.0.1:5000/.

# What to install?

pip install numpy pandas flask scikit-learn annoy librosa replicate pillow