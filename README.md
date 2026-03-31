# MLDJ

AI/ML DJ for CSE 474


**Advanced Machine Learning Models**
We will leverage three distinct models to achieve professional-grade
mixing:

• HTS-AT: An audio transformer used to classify and quantify
audio features for sound profiling.

• K-Means Clustering: Used to group songs with compatible
"vibes" and energy levels.

• Meta AudioCraft: An open source generative model used to
synthesize the actual audio transitions between songs.


**Dataset**
We will utilize human-mixed DJ sets from public sources like 1001Track-
lists and the Free Music Archive (FMA) for supervised training.


**Evaluation**
Quantitative: We will evaluate the clustering accuracy using the
Calinski-Harabasz score and measure the smoothness of transitions
by calculating the continuity of the energy curve’s derivative.
Qualitative: We expect to produce spectrogram visualizations show-
ing seamless track layering and energy alignment plots that demon-
strate a lack of jarring volume shifts.


**Computational Resources**
We will utilize Google Colab with GPU acceleration (T4/A100) to
train our Transformer and run the generative audio models. We
will also use a dedicated database to store the pre-computed sound
profiles.

@inproceedings{fma_dataset,
  title = {{FMA}: A Dataset for Music Analysis},
  author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle = {18th International Society for Music Information Retrieval Conference (ISMIR)},
  year = {2017},
  archiveprefix = {arXiv},
  eprint = {1612.01840},
  url = {https://arxiv.org/abs/1612.01840},
}
