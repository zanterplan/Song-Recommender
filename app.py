import csv
from flask import Flask, request, render_template, jsonify
from src.recommendation import *
from src.data_preprocessing import load_data
from src.extract_valence_from_image import *
from src.spotify import *
from src.features import *
import os

app = Flask(__name__)

# Load dataset
DATA_PATH = "spotify-tracks-dataset/dataset.csv"
df = load_data(DATA_PATH)

# Route: Home page
@app.route('/')
def home():
    return render_template('index.html')

# Recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get the song name from the frontend
        data = request.get_json()

        # Get song and artist data
        song = [x.strip() for x in data.get('song_name').split('-')]
        song_name = song[0]

        artists = ''
        if len(song) == 2:
            artists = song[1]
        if len(song) == 3:
            song_name = song[0] + ' - ' + song[1]
            artists = song[2]

        # Get other data
        top_n = int(data.get('top_n', 10))
        genre = data.get('genre', '').strip().lower()
        method = data.get('recommendation_method', '')
        random = data.get('random_method', False)

        # Get filters
        happiness = float(data.get('happiness', 50)) / 100
        popularity = float(data.get('popularity', 50)) / 100
        energy = float(data.get('energy', 50)) / 100
        danceability = float(data.get('danceability', 50)) / 100
        loudness = float(data.get('loudness', 50)) / 100

        # Get the entered song
        songs = df[df['track_name'].str.lower() == song_name.lower()]
        if artists != '':
            songs = songs[songs['artists'].str.lower() == artists.lower()]
        identified_song = songs.sort_values('popularity', ascending=False).iloc[0]

        # Check for randomness
        variables  = {
            'happiness': happiness,
            'energy': energy,
            'danceability': danceability,
            'loudness': loudness
        }

        # Random factor code
        if random:
            interval = np.arange(50,100,1)
            popularity /= np.random.choice(interval)

            for var_name, value in variables.items():
                random_factor = np.random.choice(interval)
                operation = np.random.choice(['multiply', 'divide'])
                if operation == 'multiply':
                    variables[var_name] *= random_factor
                elif operation == 'divide' and variables[var_name] > 0:
                    variables[var_name] /= random_factor

        # print(variables)
        happiness, energy, danceability, loudness = variables['happiness'], variables['energy'], variables['danceability'], variables['loudness']

        # Run the recommendation function
        if method in ['simple', 'advanced']:
            recommendations = recommend_songs(df, song_name, method, artists, genre, happiness, popularity, energy, danceability, loudness, top_n)
        else:
            recommendations = recommend_songs_with_annoy(df, song_name, artists, happiness, popularity, energy, danceability, loudness, top_n)
        print("Recommendations DataFrame:\n", recommendations.head(top_n))
        
        # Convert the recommendations to dictionaries
        recs = recommendations.to_dict(orient='records')
        
        # Fetch Spotify metadata for each recommendation
        metadata_recommendations = []
        token = get_spotify_token()
        for rec in recs:
            metadata = fetch_spotify_metadata(rec['track_name'], rec['artists'], token)
            if metadata:
                metadata_recommendations.append(metadata)

        # Return the recommendations
        return jsonify({
            "identified_song": {
                "name": identified_song['track_name'],
                "artist": identified_song['artists'],
                "id": identified_song['track_id']
            },
            "recommendations": metadata_recommendations})
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

# Suggestions
@app.route('/search_suggestions')
def search_suggestions():
    query = request.args.get('query', '')
    if query:
        # Get song suggestions
        suggestions = get_song_suggestions(df, query)
        # print(suggestions)
        return jsonify({'suggestions': suggestions})
    return jsonify({'suggestions': []})

genres_list = df.drop_duplicates(subset=['track_genre'])['track_genre']

# Genres
@app.route('/get_genres', methods=['GET'])
def get_genres():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])

    # Get genre suggestions
    suggestions = get_genre_suggestions(genres_list, query)
    return jsonify(suggestions)

# File upload and analyze
@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Get file and its name
    file = request.files['file']
    file_name = request.form.get('file_name')

    song = [x.strip() for x in file_name.split('-')]

    if len(song) != 2:
        print("Invalid name: " + str(len(song)))
        return None
    
    # Remove suffix from file name
    suffix = [".mp3", ".wav", ".flac"]
    for p in suffix:
        if p in song[1]:
            song[1] = song[1].replace(p, '')
    
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    # Extract song features
    valence = predict_valence_with_api(file_path)
    # valence = -1
    loudness = extract_loudness(file_path)
    danceability = extract_danceability(file_path)
    energy = extract_energy(file_path)
    popularity = get_popularity(file_name)

    # Get next row index in dataset
    next_index = -1
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            next_index = len(rows) - 1

    # Prepare data
    new_data = {
        '': next_index,
        'track_id': next_index,
        'artists': song[1],
        'album_name': '/',
        'track_name': song[0],
        'popularity': popularity,
        'duration_ms': -1,
        'explicit': 'FALSE',
        'danceability': danceability,
        'energy': energy,
        'key': -1,
        'loudness': loudness,
        'mode': -1,
        'speechiness': -1,
        'acousticness': -1,
        'instrumentalness': -1,
        'liveness': -1,
        'valence': valence,
        'tempo': 0,
        'time_signature': -1,
        'track_genre': '/'
    }

    # Define the fieldnames (columns in the CSV file)
    fieldnames = [
        '', 'track_id', 'artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 
        'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'track_genre'
    ]

    # Add the new row to CSV
    with open(DATA_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if f.tell() == 0:
            writer.writeheader()

        writer.writerow(new_data)
        f.flush()

    # Update dataset
    update_df()

    '''print(loudness)
    print(danceability)
    print(energy)
    print(popularity)
    print(type(loudness))
    print(type(danceability))
    print(type(energy))'''

    os.remove(file_path)

    return jsonify({
        'artists': song[1],
        'track_name': song[0],
        'valence': valence,
        'loudness': loudness,
        'danceability': danceability,
        'energy': energy,
        'popularity': popularity
    })

def update_df():
    global df
    df = load_data(DATA_PATH)
    print('Database updated')

if __name__ == '__main__':
    app.run(debug=True)