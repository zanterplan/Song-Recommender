import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from annoy import AnnoyIndex

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Select relevant String features
    string_features = [
        'track_id', 'track_name', 'artists', 'track_genre'
    ]

    # Select relevant numerical features
    numerical_features = [
        'popularity', 'danceability', 'energy', 'loudness',
        'acousticness', 'instrumentalness', 'valence', 'tempo'
    ]

    df = df[string_features + numerical_features]
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df

def load_annoy_index(feature_matrix, save_path="annoy_index.ann"):
    num_features = feature_matrix.shape[1]
    index = AnnoyIndex(num_features, metric='manhattan')

    if os.path.exists(save_path):
        index.load(save_path)
        # print(f"Annoy index loaded from {save_path}")
    else:
        raise FileNotFoundError(f"Annoy index file not found at {save_path}")
    
    return index

def build_and_save_annoy_index(feature_matrix, feature_weights, save_path="annoy_index.ann", num_trees=10):
    num_features = feature_matrix.shape[1]
    index = AnnoyIndex(num_features, metric='manhattan')
    
    # Add items to the index
    for i, vector in enumerate(feature_matrix):
        weighted_vector = vector * feature_weights
        index.add_item(i, weighted_vector)
    
    # Build the index
    index.build(num_trees)
    index.save(save_path)
    # print(f"Annoy index built and saved to {save_path}")
