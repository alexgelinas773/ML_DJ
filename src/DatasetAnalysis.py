import pandas as pd
import numpy as np
import librosa
import datetime
import requests

#dataset from www.kaggle.com/datasets/solomonameh/spotify-music-dataset?resource=download
#popular_songs = pd.read_csv("Songs/high_popularity_spotify_data.csv")

def skrillex2015():
    with open("DJ_Sets/skrillex2015", "r") as file:
        next(file)
        next(file)
        times = []
        for line in file:
            time_obj = datetime.datetime.strptime(line.strip().strip('[').strip(']'), "%M:%S")
            total_seconds = time_obj.minute * 60 + time_obj.second
            times.append(total_seconds)
    return times

import librosa
import numpy as np

def extract_features(audio_path, start, end):
    #Extracts core audio features from a segment
    y, sr = librosa.load(audio_path, offset=start, duration=end-start)
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # Energy
    rms = librosa.feature.rms(y=y).mean()
    # Brightness
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    # Flux
    flux = librosa.onset.onset_strength(y=y, sr=sr).mean()
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)

    return {
        "tempo": tempo,
        "rms": rms,
        "centroid": centroid,
        "flux": flux,
        **{f"mfcc_{i}": mfcc[i] for i in range(13)}
    }

def build_transition_features(df):
    transitions = []
    for i in range(len(df)-1):
        a = df.iloc[i]
        b = df.iloc[i+1]
        row = {
            "from_track": int(a.track_id),
            "to_track": int(b.track_id),
            "delta_tempo": float((b.tempo - a.tempo)[0]),
            "delta_rms": b.rms - a.rms,
            "delta_centroid": b.centroid - a.centroid,
            "delta_flux": b.flux - a.flux,
        }

        # MFCC deltas
        for k in range(13):
            row[f"delta_mfcc_{k}"] = b[f"mfcc_{k}"] - a[f"mfcc_{k}"]

        transitions.append(row)

    return pd.DataFrame(transitions)


if __name__ == '__main__':
    timestamps = skrillex2015()
    df_tracks = pd.DataFrame({
        "track_id": range(len(timestamps) - 1),
        "start": timestamps[:-1],
        "end": timestamps[1:]
    })

    audio_path = "DJ_Sets/skrillex2015audio.mp3"

    feature_rows = []
    for _, row in df_tracks.iterrows():
        feats = extract_features(audio_path, row.start, row.end)
        feats["track_id"] = row.track_id
        feature_rows.append(feats)

    df_features = pd.DataFrame(feature_rows)
    df_transitions = build_transition_features(df_features)

    df_features.to_csv("DJ_Sets/skrillex2015_features.csv")
    df_transitions.to_csv("DJ_Sets/skrillex2015_transitions.csv")

