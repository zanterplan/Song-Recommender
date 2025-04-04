import requests

# Track popularity function
def get_track_popularity(track_name, artist_name, access_token):
    # Send a request to the API
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "q": f"track:{track_name} artist:{artist_name}",
        "type": "track",
        "limit": 1
    }
    response = requests.get(url, headers=headers, params=params)

    # Find the song (if exists in the database) and get its popularity
    if response.status_code == 200:
        results = response.json()
        if results["tracks"]["items"]:
            track = results["tracks"]["items"][0]
            return {
                "popularity": track["popularity"]
            }
    return None

# Metadata and token init
SPOTIFY_CLIENT_ID = 'aa4e233d839f4e248f23df61ed64b5b6'
SPOTIFY_CLIENT_SECRET = '0e5323f5e22444f6bbb936da3410299b'

def fetch_spotify_metadata(track_name, artist_name, access_token):
    # Send a request to the API
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "q": f"track:{track_name} artist:{artist_name}",
        "type": "track",
        "limit": 1
    }
    response = requests.get(url, headers=headers, params=params)

    # Find the song (if exists in the database) and get the data for cards
    if response.status_code == 200:
        results = response.json()
        if results["tracks"]["items"]:
            track = results["tracks"]["items"][0]
            return {
                "title": track["name"],
                "artist": ", ".join([artist["name"] for artist in track["artists"]]),
                "image_url": track["album"]["images"][0]["url"] if track["album"]["images"] else None
            }
    return None

def get_spotify_token():
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials"
    }
    response = requests.post(url, headers=headers, data=data, auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET))
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        raise Exception("Failed to get Spotify token: " + response.text)