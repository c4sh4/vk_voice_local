# imports
import os
import librosa
import pandas as pd
from tqdm import tqdm
from google.colab import drive
from IPython.display import Audio

# """
# RAVDESS explain:
# 03-01-06-01-02-01-12.wav
# modality = (01 = full-AV, 02 = video-only, 03 = audio-only)
# vocal_channel = (01 = speech, 02 = song)
# emotion:
#
# 01 = neutral,
# 02 = calm,
# 03 = happy,
# 04 = sad,
# 05 = angry,
# 06 = fearful,
# 07 = disgust,
# 08 = surprised
# emotional intensity = (01 = normal, 02 = strong)
# statement, repetiotion, actor.
# """


def load_and_resample_wav(path: str, sample_rate: int = 0) -> pd.DataFrame:
    """
    load and resample audio files from given path
    """
    df = pd.DataFrame(columns=['audio', 'label'])
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            parts = filename.split('-')
            if parts[0] == '03' and parts[1] == '01' and parts[2] in ['01', '02', '03', '04', '05', '06', '07', '08'] and parts[3] in ['01', '02']:
                #  exctract emotion label
                emotion = int(filename.split('-')[2])-1

                # load and resample wav
                y, _ = librosa.load(os.path.join(path, filename), sr=48000)
                if sample_rate != 0:
                  y = librosa.resample(y=y, orig_sr=48000, target_sr=sample_rate)
                df.loc[len(df)] = {'audio': y, 'label': emotion}
    return df


def create_RAVDESS_df_with_labels(RAVDESS_path: str, sample_rate: int = 0) -> pd.DataFrame:
    df = pd.DataFrame(columns=['audio', 'label'])
    for actor in tqdm(os.listdir(RAVDESS_path)):
        actor_path = os.path.join(RAVDESS_path, actor)
        if actor.startswith("Actor") and os.path.isdir(actor_path):
            temp_df = load_and_resample_wav(actor_path, sample_rate)
            df = pd.concat([df, temp_df], ignore_index=True)
    return df
