import torch
import json
import numpy as np
import torchaudio
from hifi_gan_models import Generator  # your local Generator file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Paths
# =========================
SOURCE_MEL_PATH = "./prototyping/preprocessed/chunk_030.npy"
HIFIGAN_CKPT = "./prototyping/models/g_03280000"  # pretrained weights
HIFIGAN_CONFIG = "./prototyping/models/config_v1.json"
OUTPUT_WAV = "./prototyping/converted.wav"

# =========================
# Load mel
# =========================
mel = np.load(SOURCE_MEL_PATH)  # (T, n_mels)
mel = torch.tensor(mel.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, n_mels, T)

# =========================
# Load HiFi-GAN config
# =========================
with open(HIFIGAN_CONFIG) as f:
    config = json.load(f)

# The Generator class expects a single object with attributes
class HifiGanConfig:
    def __init__(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)

h = HifiGanConfig(config)

# =========================
# Create generator
# =========================
generator = Generator(h).to(DEVICE)

# =========================
# Load pretrained weights
# =========================
checkpoint = torch.load(HIFIGAN_CKPT, map_location=DEVICE)
# Check for wrapped checkpoint
if "generator" in checkpoint:
    state_dict_g = checkpoint["generator"]
else:
    state_dict_g = checkpoint

generator.load_state_dict(state_dict_g)
generator.eval()

# =========================
# Generate waveform
# =========================
with torch.no_grad():
    waveform = generator(mel)           # (1, 1, T)
    waveform = waveform.squeeze(0)      # (1, T)
    waveform = waveform.cpu()           # move to CPU

torchaudio.save(OUTPUT_WAV, waveform, sample_rate=16000)



# =========================
# Save waveform
# =========================
# torchaudio.save(OUTPUT_WAV, waveform.unsqueeze(0), sample_rate=16000)
print(f"Audio saved to {OUTPUT_WAV}")
