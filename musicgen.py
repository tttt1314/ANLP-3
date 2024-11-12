import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-medium')
model.set_generation_params(duration=10)  # generate 8 seconds.

musiccaps_df = pd.read_csv('./musiccaps-public.csv')
captions = musiccaps_df['caption'].values
ids = musiccaps_df['ytid'].values

wav = model.generate(captions[:2])  # generates 2 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    name = ids[idx]
    audio_write(name, one_wav.cpu(), model.sample_rate, strategy="loudness")