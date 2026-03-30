import pandas as pd
import requests

popular_songs = pd.read_csv("Songs/high_popularity_spotify_data.csv")

if __name__ == '__main__':
    print(popular_songs[:5])