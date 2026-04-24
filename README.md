# README: Automating the DJ Experience

Authors: Alexandra Gelinas, Sammy Odeh, Jaidin Rosario

# TLDR

Automated analysis and evaluation of DJ set transitions using audio feature extraction and machine learning (neural network). Our project trains a ML Model using features like MFCCs, RMS energy, and spectral flux from live DJ Sets extracted from 1001Tracklists and MixesDB. This project utilizes libraries mentioned throughout the ReadMe in order to create a project purely for educational purposes for our course CSE 474: Intro to Machine Learning. Through this project, we were able to apply concepts learned as class such as what supervised learning is, neural networks, vector projection, etc.

## Project Structure

```
.
├── Bad/                        # Examples of poorly executed transitions
├── DataInputs/                 # .txt files of timestamps and .mp3 audio files of DJ sets
├── Datasets/                   # Datasets containing extracted feature and transition data
├── audio/                      # Audio files (.mp3) from FreeMusicArchive to generate mixes
│
├── DataAnalysis.ipynb          # Initial research on song data vs danceability scores
├── DEMO_TRANSITION.wav         # Sample transition created by our model using Librosa and soundfile
├── dj.ipynb                    # ML Model. Uses extracted data, trains model, and generates mixes
├── GetTrainingData.py          # Script to get positive training data, Librosa ran to extract features
│
├── high_popularity_spotify_data.csv   # Spotify dataset from Kaggle
├── phrase_ends.npy             # NumPy array of detected phrase end timestamps
├── phrase_features.npy         # NumPy array of extracted features per phrase
├── phrase_files.npy            # NumPy array mapping phrases to source audio files
├── phrase_starts.npy           # NumPy array of detected phrase start timestamps
├── skrillex2015                # Reference DJ set (Skrillex 2015 mix) - First one we extracted!
│
├── .gitignore
└── README.md                    # you are here :)
```

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. **Install dependencies:**

   ```bash
   pip install numpy pandas librosa matplotlib jupyter soundfile
   ```

3. **Launch Jupyter and explore:**

   ```bash
   jupyter notebook
   ```

   Run `dj.ipynb` for transition analysis and to generate mixes!

   P.S. We wanted to explore musical data because we all love music and are interested in creative computional work.
