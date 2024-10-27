import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request
import numpy as np
import random

app = Flask(__name__)

file_path = '/app/Book1.xlsx'  # This should match the path in the Docker container
data = pd.read_excel(file_path)


# Step 1: Select relevant features
features = ['energy_%', 'danceability_%', 'acousticness_%']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])

# Step 2: KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Define moods for clusters
cluster_moods = {0: 'Energetic', 1: 'Chill', 2: 'Acoustic'}

# Q-Learning Playlist Class
class QLearningPlaylist:
    def __init__(self, data, cluster, n_songs=15):
        self.data = data
        self.cluster = cluster
        self.n_songs = n_songs
        self.q_table = {}  # Initialize Q-table for state-action pairs

    def initialize_q_table(self):
        for song_id in self.data.index:
            self.q_table[song_id] = random.uniform(0, 1)  # Random Q-value initialization

    def update_q_value(self, song_id, reward, learning_rate=0.1, discount_factor=0.95):
        old_q_value = self.q_table[song_id]
        max_next_q = max(self.q_table.values())
        self.q_table[song_id] = old_q_value + learning_rate * (reward + discount_factor * max_next_q - old_q_value)

    def generate_playlist(self):
        self.initialize_q_table()
        playlist_songs = []
        selected_songs = set()

        for _ in range(self.n_songs):
            cluster_songs = [song_id for song_id in self.data[self.data['Cluster'] == self.cluster].index if song_id not in selected_songs]
            if not cluster_songs:
                break
            best_song_id = max(cluster_songs, key=lambda x: self.q_table[x])
            playlist_songs.append(best_song_id)
            selected_songs.add(best_song_id)

            feedback = random.choice([1, -1])
            self.update_q_value(best_song_id, reward=feedback)

        playlist = self.data.loc[playlist_songs][['track_name', 'artist(s)_name']]
        return playlist

# Policy Gradient Playlist Class
class PolicyGradientPlaylist:
    def __init__(self, data, cluster, n_songs=15):
        self.data = data
        self.cluster = cluster
        self.n_songs = n_songs

    def generate_playlist(self):
        cluster_songs = self.data[self.data['Cluster'] == self.cluster]
        probabilities = np.random.dirichlet(np.ones(len(cluster_songs)), size=1)[0]
        song_ids = cluster_songs.index

        selected_songs = np.random.choice(song_ids, size=self.n_songs, replace=False, p=probabilities)
        playlist = self.data.loc[selected_songs][['track_name', 'artist(s)_name']]
        return playlist

@app.route('/')
def index():
    return render_template('index.html', moods=cluster_moods)

@app.route('/generate_playlist', methods=['POST'])
def generate_playlist():
    user_input_cluster = int(request.form['cluster'])
    mood = cluster_moods.get(user_input_cluster, "Unknown Mood")

    policy_gradient = PolicyGradientPlaylist(data, user_input_cluster)
    policy_gradient_playlist = policy_gradient.generate_playlist()

    return render_template(
        'playlist.html',
        mood=mood,
        policy_gradient_playlist=policy_gradient_playlist.to_dict(orient='records'),
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

