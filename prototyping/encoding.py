import os
import torch
import torchaudio
import numpy as np
from speechbrain.inference import EncoderClassifier  # updated import

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_WAV = "./prototyping/audio/lena.wav"
OUTPUT_PATH = "./prototyping/encoded"
TARGET_SR = 16000  # SpeechBrain standard

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ======================
# Load SpeechBrain ECAPA-TDNN model
# ======================
speaker_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

# ======================
# Load WAV and resample if needed
# ======================
signal, sr = torchaudio.load(INPUT_WAV)

if sr != TARGET_SR:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
    signal = resampler(signal)
    sr = TARGET_SR

signal = signal.to(DEVICE)

# ======================
# Encode speaker
# ======================
with torch.no_grad():
    embedding = speaker_encoder.encode_batch(signal)

embedding = embedding.squeeze(0).cpu().numpy()

# ======================
# Save embedding
# ======================
out_file = os.path.join(OUTPUT_PATH, "speaker_embedding.npy")
np.save(out_file, embedding)

print(f"Encoding completed. Speaker embedding saved to: {out_file}")
