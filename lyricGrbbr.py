import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
from bs4 import BeautifulSoup
import json
import csv


# Spotify API credentials
SPOTIPY_CLIENT_ID = 'a3d9f70f838a466d85d89cec9cde35eb'
SPOTIPY_CLIENT_SECRET = '3b2db8d3c4aa43bfb004fe8888391d73'
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'

# Genius API credentials
GENIUS_ACCESS_TOKEN = 'fF2yBqabkNqiZmURRMLi-Ue73DH5DWHm4jVz5HVyPWNu6D7OGlmdTMncmtfPIDOv'

# Authenticate with Spotify
scope = "user-top-read"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope=scope))

def get_top_tracks():
    results = sp.current_user_top_tracks(limit=20)
    tracks = results['items']
    return [(track['name'], track['artists'][0]['name']) for track in tracks]

def get_lyrics(song_title, artist_name):
    base_url = "https://api.genius.com"
    headers = {'Authorization': 'Bearer ' + GENIUS_ACCESS_TOKEN}
    search_url = base_url + "/search"
    data = {'q': song_title + ' ' + artist_name}
    response = requests.get(search_url, data=data, headers=headers)
    
    if response.status_code != 200:
        print(f"Error fetching search results for {song_title} by {artist_name}: {response.status_code}")
        return None

    json_data = response.json()
    print(f"Search results for {song_title} by {artist_name}:", json.dumps(json_data, indent=4))

    if json_data['response']['hits']:
        song_api_path = json_data['response']['hits'][0]['result']['api_path']
        song_url = base_url + song_api_path
        song_response = requests.get(song_url, headers=headers)
        song_json = song_response.json()
        print(f"Song details for {song_title} by {artist_name}:", json.dumps(song_json, indent=4))

        if 'response' in song_json and 'song' in song_json['response']:
            song_info = song_json['response']['song']
            if 'url' in song_info:
                lyrics_url = song_info['url']
                lyrics = scrape_lyrics(lyrics_url)
                return lyrics
            else:
                print(f"Lyrics URL not found for {song_title} by {artist_name}")
        else:
            print(f"Unexpected structure for {song_title} by {artist_name}")
    
    return None

def scrape_lyrics(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    # Look for the div that contains the lyrics
    lyrics_div = soup.find('div', class_='lyrics') or soup.find('div', class_='Lyrics__Root-sc-1ynbvzw-0')
    if lyrics_div:
        # Extract the text and split it into lines
        lyrics = lyrics_div.get_text(separator='\n').strip().split('\n')
        # Filter out lines that are not part of the lyrics
        filtered_lyrics = [line for line in lyrics if not line.startswith('[') and not 'Contributor' in line]
        lyrics_text = '\n'.join(filtered_lyrics)

        # Replace "Embed" at the end with "[CLS]"
        if lyrics_text.endswith("Embed"):
            lyrics_text = lyrics_text[:-5] + "[CLS]"

        return lyrics_text
    else:
        print(f"Lyrics not found on page: {url}")
        return None


def save_lyrics_to_file(lyrics_data):
    with open('song_lyrics.csv', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Title', 'Artist', 'Lyrics'])  # Write the header
        for (title, artist), lyrics in lyrics_data.items():
            writer.writerow([title, artist, lyrics])


def main():
    top_tracks = get_top_tracks()
    print("Top tracks:", top_tracks)  # Debugging output
    lyrics_data = {}
    
    for title, artist in top_tracks:
        lyrics = get_lyrics(title, artist)
        if lyrics:
            lyrics_data[(title, artist)] = lyrics
        else:
            print(f"Lyrics not found for {title} by {artist}")
    
    save_lyrics_to_file(lyrics_data)
    print(f"Lyrics data saved to file: {lyrics_data}")  # Debugging output

if __name__ == "__main__":
    main()
