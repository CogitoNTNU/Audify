import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
DATASET_PATH = "./prototyping/dataset/VCTK-Corpus/wav48"
OUTPUT_PATH = "prototyping/preprocessed"
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
TRIM_SILENCE = True
SAVE_IMAGES = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# TORCHAUDIO MEL TRANSFORM (GPU)
# =========================
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
).to(DEVICE)

def load_wav(file_path, sr=SAMPLE_RATE, trim_silence=TRIM_SILENCE):
    y, orig_sr = torchaudio.load(file_path)
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        y = resampler(y)
    y = y.squeeze(0)  # mono

    if trim_silence:
        y_vad = torchaudio.functional.vad(torch.tensor(y).unsqueeze(0), sample_rate=sr)
        y_vad = y_vad.squeeze(0).numpy()
        if len(y_vad) == 0:
            # fallback: skip trimming if VAD removed all audio
            y_trimmed = y.numpy() if isinstance(y, torch.Tensor) else y
            return y_trimmed
        return y_vad

    return y.numpy() if isinstance(y, torch.Tensor) else y



def wav_to_mel(y):
    y_tensor = torch.tensor(y, device=DEVICE).unsqueeze(0)
    mel = mel_transform(y_tensor)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel.squeeze(0).cpu().numpy().T  # Time x Mel

def save_wave_and_mel_image(y, mel, save_path, sr=SAMPLE_RATE):
    plt.figure(figsize=(12, 6))
    # Waveform
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, len(y)/sr, len(y)), y, color='steelblue')
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    # Mel-spectrogram
    plt.subplot(2, 1, 2)
    plt.imshow(mel.T, origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel-Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bins")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =========================
# PROCESS DATASET
# =========================
speakers = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

for spk in speakers:
    spk_path = os.path.join(DATASET_PATH, spk)
    out_spk_path = os.path.join(OUTPUT_PATH, spk)
    os.makedirs(out_spk_path, exist_ok=True)
    img_path = os.path.join(out_spk_path, "images")
    if SAVE_IMAGES:
        os.makedirs(img_path, exist_ok=True)

    wav_files = glob(os.path.join(spk_path, "*.wav"))
    for wav_file in tqdm(wav_files, desc=f"Processing {spk}"):
        y = load_wav(wav_file)
        mel = wav_to_mel(y)

        # Save mel spectrogram as numpy array
        mel_name = os.path.splitext(os.path.basename(wav_file))[0] + ".npy"
        np.save(os.path.join(out_spk_path, mel_name), mel)

        # Save waveform + mel image
        if SAVE_IMAGES:
            img_name = os.path.splitext(os.path.basename(wav_file))[0] + ".png"
            save_wave_and_mel_image(y, mel, os.path.join(img_path, img_name))

print("Preprocessing completed. Mel-spectrograms and images saved to:", OUTPUT_PATH)
