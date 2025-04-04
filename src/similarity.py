import numpy as np
from sklearn.preprocessing import normalize
from annoy import AnnoyIndex
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize, StandardScaler
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans

def compute_simple_similarity(feature_matrix, query_vector, feature_weights=None):
    # Default equal weights
    if feature_weights is None:
        feature_weights = np.ones(feature_matrix.shape[1])
    else:
        feature_weights = np.array(feature_weights)
    
    # Standardize the features
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    query_vector = scaler.transform(query_vector.reshape(1, -1)).flatten()

    # Apply exponential weighting
    weighted_matrix = feature_matrix * np.exp(feature_weights)
    weighted_query = query_vector * np.exp(feature_weights)

    # Multiply combinations of features
    feature_interactions = np.array([
        weighted_matrix[:, i] * weighted_matrix[:, j] 
        for i in range(weighted_matrix.shape[1])
        for j in range(i+1, weighted_matrix.shape[1])
    ]).T
    
    weighted_matrix = np.hstack([weighted_matrix, feature_interactions])
    weighted_query = np.hstack([weighted_query, 
                                [weighted_query[i] * weighted_query[j] for i in range(weighted_query.shape[0])
                                 for j in range(i+1, weighted_query.shape[0])]])

    # Normalize
    weighted_matrix = normalize(weighted_matrix, axis=1)
    weighted_query = normalize(weighted_query.reshape(1, -1), axis=1)

    # Compute cosine similarity
    cosine_scores = np.dot(weighted_matrix, weighted_query.T).flatten()

    # Add feature-based corrections
    transformed_matrix = np.copy(weighted_matrix)
    transformed_matrix[:, 0] = np.log(1 + transformed_matrix[:, 0])
    transformed_matrix[:, 1] = 1 / (1 + np.exp(-transformed_matrix[:, 1]))

    # Normalize transformed matrix
    transformed_matrix = normalize(transformed_matrix, axis=1)

    # Cosine similarity
    transformed_scores = np.dot(transformed_matrix, weighted_query.T).flatten()

    # Penalize songs with extremely high similarity scores
    penalty = np.exp(-np.abs(cosine_scores - transformed_scores))
    final_scores = cosine_scores * penalty

    final_similarity_scores = 0.5 * final_scores + 0.3 * transformed_scores + 0.2 * cosine_scores

    return final_similarity_scores

def compute_advanced_similarity(feature_matrix, query_vector, feature_weights=None):    
    # Standardize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    query_vector_scaled = scaler.transform(query_vector.reshape(1, -1)).flatten()
    
    # Exponential weighting (better for important features)
    weighted_matrix = feature_matrix_scaled * np.exp(feature_weights)
    weighted_query = query_vector_scaled * np.exp(feature_weights)
    
    # Interaction terms (cross-product of features)
    interaction_matrix = np.column_stack([weighted_matrix, weighted_matrix ** 2, weighted_matrix ** 3])
    interaction_query = np.concatenate([weighted_query, weighted_query ** 2, weighted_query ** 3])
    
    # Normalize interaction features
    interaction_matrix = MinMaxScaler().fit_transform(interaction_matrix)
    interaction_query = MinMaxScaler().fit_transform(interaction_query.reshape(1, -1)).flatten()
    
    # Non-linear similarity (Radial Basis Function (RBF) kernel)
    rbf_similarity = compute_rbf_similarity(weighted_matrix, weighted_query)
    
    # Cosine similarity
    cosine_sim = cosine_similarity(weighted_matrix, weighted_query.reshape(1, -1)).flatten()
    
    # Clustering-based similarity
    clustering_sim = compute_clustering_similarity(feature_matrix_scaled, weighted_query)
    
    # Final similarity score
    final_similarity = 0.5 * cosine_sim + 0.25 * rbf_similarity + 0.25 * clustering_sim
    
    return final_similarity

def compute_rbf_similarity(feature_matrix, query_vector, gamma=0.1):
    # Pairwise distance and apply RBF kernel
    distances = np.linalg.norm(feature_matrix - query_vector, axis=1)
    rbf_kernel = np.exp(-gamma * distances ** 2)
    return rbf_kernel

def compute_clustering_similarity(feature_matrix, query_vector, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(feature_matrix)
    
    # Nearest cluster centroid to the query song
    cluster_label = kmeans.predict([query_vector])[0]
    cluster_centroid = kmeans.cluster_centers_[cluster_label]
    
    # Compute distance between them
    cluster_distance = np.linalg.norm(cluster_centroid - query_vector)
    return np.exp(-cluster_distance)

'''def simple_similarity(feature_matrix, query_vector, feature_weights=None):
    # Default equal weights
    if feature_weights is None:
        feature_weights = np.ones(feature_matrix.shape[1])
    else:
        feature_weights = np.array(feature_weights)

    # Standardize the features
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    query_vector = scaler.transform(query_vector.reshape(1, -1)).flatten()
    
    # Apply weights
    weighted_matrix = feature_matrix * np.exp(feature_weights)
    weighted_query = query_vector * np.exp(feature_weights)
    
    # Normalize vectors
    weighted_matrix = normalize(weighted_matrix, axis=1)
    weighted_query = normalize(weighted_query.reshape(1, -1), axis=1)

    # Compute cosine similarity
    cosine_scores = np.dot(weighted_matrix, weighted_query.T).flatten()
    
    # Euclidean similarity (weighted)
    euclidean_scores = np.linalg.norm(weighted_matrix - weighted_query, axis=1)
    euclidean_scores = 1 / (1 + euclidean_scores)  # Convert distances to similarities

    # Combine similarities
    blended_scores = 0.7 * cosine_scores + 0.3 * euclidean_scores

    return blended_scores'''

def get_annoy_recommendations(index, df, query_vector, feature_weights, top_n=16):
    weighted_query_vector = query_vector * feature_weights

    # Annoy index for nearest neighbors
    nearest_indices = index.get_nns_by_vector(weighted_query_vector, top_n + 1, include_distances=False)

    # Exclude the first result if it's the same as the query song
    recommended_songs = df.iloc[nearest_indices]
    return recommended_songs