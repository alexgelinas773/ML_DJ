import pandas as pd
import numpy as np
import librosa
import datetime
import requests

#dataset from www.kaggle.com/datasets/solomonameh/spotify-music-dataset?resource=download
#popular_songs = pd.read_csv("Songs/high_popularity_spotify_data.csv")
#skrillex 2015 set from https://soundcloud.com/umfmusic-1/jack-u-ultra-music-festival-2015
#van Buren 2026 set from https://www.youtube.com/watch?v=1wzzDJtYNFI
#garrixThailand2026 from https://soundcloud.com/tech-noise-1456671696/martin-garrix-live-siam?utm_source=clipboard&utm_campaign=wtshare&utm_medium=widget&utm_content=https%253A%252F%252Fsoundcloud.com%252Ftech-noise-1456671696%252Fmartin-garrix-live-siam

def getTimestaps(filename):
    with open(filename, "r") as file:
        next(file)
        next(file)
        times = []
        for line in file:
            time_obj = datetime.datetime.strptime(line.strip().strip('[').strip(']'), "%H:%M:%S")
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

def main(track_stimestamp_name, audio_file_name):
    track_timestamp_file = f"DataInputs/{track_stimestamp_name}"
    audio_path = f"DataInputs/{audio_file_name}.mp3"

    timestamps = getTimestaps(track_timestamp_file)
    #print(timestamps)
    df_tracks = pd.DataFrame({
        "track_id": range(len(timestamps) - 1),
        "start": timestamps[:-1],
        "end": timestamps[1:]
    })

    feature_rows = []
    for _, row in df_tracks.iterrows():
        feats = extract_features(audio_path, row.start, row.end)
        feats["track_id"] = row.track_id
        feature_rows.append(feats)

    df_features = pd.DataFrame(feature_rows)
    df_transitions = build_transition_features(df_features)

    df_features.to_csv(f"Datasets/{audio_file_name}_features.csv")
    df_transitions.to_csv(f"Datasets/{audio_file_name}_transitions.csv")

def bad_transition(timestamps, audio_file_name):
    audio_path = f"DataInputs/{audio_file_name}.mp3"

    # If only [start, end] is given, split at midpoint to get 2 segments
    if len(timestamps) == 2:
        mid = (timestamps[0] + timestamps[1]) / 2
        timestamps = [timestamps[0], mid, timestamps[1]]

    df_tracks = pd.DataFrame({
        "track_id": range(len(timestamps) - 1),
        "start": timestamps[:-1],
        "end": timestamps[1:]
    })

    feature_rows = []
    for _, row in df_tracks.iterrows():
        feats = extract_features(audio_path, row.start, row.end)
        feats["track_id"] = row.track_id
        feature_rows.append(feats)

    df_features = pd.DataFrame(feature_rows)
    df_transitions = build_transition_features(df_features)

    df_features.to_csv(f"Bad/{audio_file_name}_features.csv")
    df_transitions.to_csv(f"Bad/{audio_file_name}_transitions.csv")

if __name__ == '__main__':
    # Put song and transition timestamp txt in /DataInputs
    # Don't include /DataInputs/ only the file name without .txt/.mp3/.wav or wtv

    track_timestamp_list = "garrixThailand2026" # don't include txt
    audio_file_name = "garrixThailand2026"# don't include mp3

    #main(track_timestamp_list, audio_file_name)
    bad_transition([220,300], "countryPopMix")
