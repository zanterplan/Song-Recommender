from flask import jsonify
import librosa
import numpy as np
import replicate
import os
from src.extract_valence_from_image import *
from src.spotify import *
from PIL import Image

# Popularity
def get_popularity(file_name):
    song = [x.strip() for x in file_name.split('-')]

    # Check if it's in the correct format
    if len(song) != 2:
        print("Invalid length: " + str(len(song)))
        return -1
    
    # Remove suffix from the song
    suffix = [".mp3", ".wav", ".flac"]
    for p in suffix:
        if p in song[1]:
            song[1] = song[1].replace(p, '')

    token = get_spotify_token()
    popularity = get_track_popularity(song[0], song[1], token)

    return 0 if popularity is None else popularity["popularity"]

# Not relevant
def extract_loudness1(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    rms = librosa.feature.rms(y=y)
    loudness = rms.mean()

    pre_min, pre_max = 0, 0.3
    normalized_loudness = (loudness - pre_min) / (pre_max - pre_min)

    return float(normalized_loudness)

def extract_loudness(file_path):
    y, sr = librosa.load(file_path, sr=None)
    S = np.abs(librosa.stft(y))**2
    psd = np.mean(S, axis=1)
    loudness_curve = np.array([1.0 if f < 10 else f**-0.5 for f in np.linspace(10, sr//2, len(psd))])
    weighted_psd = psd * loudness_curve
    perceived_rms = np.sqrt(np.sum(weighted_psd))

    # Normalize
    normalized_loudness = -20 + (perceived_rms / 75) * (18)

    return float(normalized_loudness)

def extract_danceability(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)

    # Calculate tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # Normalize between 50 db and 210 db
    danceability = (tempo[0] - 50) / (210 - 50)
    danceability = max(0, min(1, danceability))

    return float(danceability)

def extract_energy(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Root-mean-square (RMS)
    rms_energy = librosa.feature.rms(y=y)
    mean_energy = np.mean(rms_energy)

    # Normalize
    m = (mean_energy - 0.0) / (0.35 - 0.0)
    m = max(0, min(1, m))

    return float(m)

# Valence API integration - pretrained model
def predict_valence_with_api(audio_file_path):
    # Call API
    with open(audio_file_path, "rb") as audio_file:
        output = replicate.run(
            "mtg/music-arousal-valence:478e0c3cf5b06b020089f904bfb29f235e4bcc29afa8a511337f15a7632978c5",
            input={"audio": audio_file, "dataset": "deam", "embedding_type": "msd-musicnn"},
        )

    print(f"API Output: {output}")
    
    try:
        # Extract valence from image
        with open("uploads/out.png", "wb") as file:
            file.write(output.read())
        print(f"Image saved as 'uploads/out.png'")
        valence, arousal = extract_valence_arousal_from_graph(Image.open("uploads/out.png"))
        os.remove("uploads/out.png")
        return valence
    except Exception as e:
        print(f"Error saving image: {e}")
    return -5

# Pretrained model (attemp for valence)
'''def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    
    n_mels = 187
    target_frames = 96
    hop_length = len(y) // (target_frames - 1)

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram.astype(np.float32)

def load_frozen_model(pb_file_path):
    with tf.io.gfile.GFile(pb_file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

def predict_valence(audio_features):
    d_model = load_frozen_model("models/valence/deam-msd-musicnn-2.pb")

    input_tensor = d_model.get_tensor_by_name("model/Placeholder:0")
    output_tensor = d_model.get_tensor_by_name("model/Identity:0")

    with tf.compat.v1.Session(graph=d_model) as sess:
        predictions = sess.run(output_tensor, feed_dict={input_tensor: [audio_features]})
        valence = float(predictions[0][0])
        return valence

def extract_features_from_audio(audio_features):
    fe_model = load_frozen_model("models/valence/msd-musicnn-1.pb")

    input_tensor = fe_model.get_tensor_by_name("model/Placeholder:0")
    output_tensor = fe_model.get_tensor_by_name("model/dense/BiasAdd:0")
    audio_features = np.expand_dims(audio_features, axis=0)
    
    with tf.compat.v1.Session(graph=fe_model) as sess:
        features = sess.run(output_tensor, feed_dict={input_tensor: audio_features})
    
    return features[0]'''

'''def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=200)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = mel_spectrogram.mean(axis=1)
    return mel_spectrogram.astype(np.float32)'''