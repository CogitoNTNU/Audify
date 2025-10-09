import torch
import torch.nn as nn
import torchaudio
import numpy as np
from vocoder import mel_to_waveform  # your existing Griffin-Lim vocoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load latent content and speaker embedding
# =========================
latent_content = np.load("./prototyping/preprocessed/chunk_030.npy")  # (T, C)
latent_content = torch.tensor(latent_content, dtype=torch.float32).to(DEVICE)

speaker_emb = np.load("./prototyping/encoded/speaker_embedding.npy")  # (D,) or (1, D)
speaker_emb = torch.tensor(speaker_emb, dtype=torch.float32).to(DEVICE)

T, C = latent_content.shape
# Ensure speaker embedding is 1D
if speaker_emb.dim() > 1:
    speaker_emb = speaker_emb.squeeze(0)
D = speaker_emb.shape[0]

# Repeat speaker embedding along time axis
speaker_emb_exp = speaker_emb.unsqueeze(0).repeat(T, 1)  # (T, D)

# Concatenate latent content + speaker embedding
decoder_input = torch.cat([latent_content, speaker_emb_exp], dim=1)  # (T, C+D)
print("Decoder input shape:", decoder_input.shape)

# =========================
# Simple linear decoder
# =========================
class LinearDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

decoder = LinearDecoder(input_dim=C+D, output_dim=C).to(DEVICE)

# =========================
# Forward pass
# =========================
with torch.no_grad():
    converted_mel = decoder(decoder_input)

print("Converted mel shape:", converted_mel.shape)  # (T, C)

# =========================
# Mel -> waveform
# =========================
waveform = mel_to_waveform(converted_mel.cpu().numpy(), sr=16000, n_fft=1024, hop_length=256)

# =========================
# Save output
# =========================
output_wav_path = "./prototyping/converted.wav"
torchaudio.save(output_wav_path, waveform.unsqueeze(0), sample_rate=16000)
print(f"Converted audio saved to: {output_wav_path}")
