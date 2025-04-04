import pandas as pd
from sklearn.decomposition import PCA
from .similarity import *
from .data_preprocessing import *

def recommend_songs(df, song_name, method='simple', artists='', genre='', happiness=0.5, popularity=0.5,
                    energy=0.5, danceability=0.5, loudness=0.5, top_n=8):
    # Remove duplicates
    removed_duplicates = df.drop_duplicates(subset=['track_name', 'artists'])
    df = removed_duplicates.drop_duplicates(subset='track_id')

    # List of names of numerical features
    numerical_features = [
        'popularity', 'danceability', 'energy', 'loudness', 'valence', 'tempo'
    ]

    # Assign weights
    feature_weights = [popularity, danceability, energy, loudness, happiness, 0.5]

    # Filter by song name and by artists
    songs = df[df['track_name'].str.lower() == song_name.lower()]
    if songs.empty:
        raise ValueError(f"Song '{song_name}' not found in the dataset.")
    if artists != '':
        songs = songs[songs['artists'].str.lower() == artists.lower()]

    # In case of multiples, select the most popular
    song = songs.sort_values('popularity', ascending=False).iloc[0]

    # Extract the feature vector of the input song
    query_vector = song[numerical_features].values
    
    # Compute similarity
    feature_matrix = df[numerical_features].values
    if method == 'simple':
        df['similarity'] = compute_simple_similarity(feature_matrix, query_vector, feature_weights)
    else:
        df['similarity'] = compute_advanced_similarity(feature_matrix, query_vector, feature_weights)
    
    # Filter and rank the most similar songs
    if genre == '':
        recommendations = (
            df[df['track_name'].str.lower() != song_name.lower()]
            .sort_values('similarity', ascending=False)
            .head(top_n)
        )
    else:
        recommendations = (
            df[
                (df['track_name'].str.lower() != song_name.lower()) &
                (df['track_genre'].str.lower() == genre.lower())
            ]
            .sort_values('similarity', ascending=False)
            .head(top_n)
        )

    return recommendations[['track_name', 'artists', 'similarity', 'track_genre']]

def get_song_suggestions(df, query):
    # Remove duplicates
    removed_duplicates = df.drop_duplicates(subset=['track_name', 'artists'])
    df = removed_duplicates.drop_duplicates(subset='track_id')
    
    query = query.lower()

    # Filter songs based on the input
    filtered_songs = df[
        df['track_name'].str.lower().str.contains(query) |
        df['artists'].str.lower().str.contains(query)
    ]

    return filtered_songs.sort_values('popularity', ascending=False).head(10)[['track_name', 'artists']].to_dict(orient='records')

def get_genre_suggestions(genres, query):
    query_lower = query.lower()
    suggestions = [genre for genre in genres if query_lower in genre.lower()]
    return suggestions

def recommend_songs_with_annoy(df, song_name, artists='', happiness=0.5, popularity=0.5,
                    energy=0.5, danceability=0.5, loudness=0.5, top_n=8, index_path="annoy_index.ann"):
    # Define numerical values and weights
    numerical_features = [
        'popularity', 'danceability', 'energy', 'loudness', 'valence', 'tempo'
    ]
    feature_weights = [popularity, danceability, energy, loudness, happiness, 0.5]

    # Remove duplicates
    removed_duplicates = df.drop_duplicates(subset=['track_name', 'artists'])
    df = removed_duplicates.drop_duplicates(subset='track_id')

    # Filter by song name and by artists
    songs = df[df['track_name'].str.lower() == song_name.lower()]
    if songs.empty:
        raise ValueError(f"Song '{song_name}' not found in the dataset.")
    if artists != '':
        songs = songs[songs['artists'].str.lower() == artists.lower()]
    
    # In case of muptiples, select the most popular
    song = songs.sort_values('popularity', ascending=False).iloc[0]

    # Extract the feature vector for the input song
    query_vector = song[numerical_features].values

    # Build or load Annoy index
    if not os.path.exists(index_path):
        feature_matrix = df[numerical_features].values
        build_and_save_annoy_index(feature_matrix, feature_weights, save_path=index_path, num_trees=50)
    
    index = load_annoy_index(df[numerical_features].values, save_path=index_path)

    # Get recommendations
    recommendations = get_annoy_recommendations(index, df, query_vector, feature_weights, top_n)
    
    # Exclude the input song itself from recommendations
    recommendations = recommendations[recommendations['track_name'].str.lower() != song_name.lower()]
    
    return recommendations[['track_name', 'artists', 'popularity', 'track_genre']].head(top_n)