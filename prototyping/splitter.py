import os
import torchaudio

# =========================
# CONFIGURATION
# =========================
INPUT_FILE = "./prototyping/dataset/audio/vol6.wav"   # your semi-long audio file
OUTPUT_PATH = "./prototyping/dataset/audio/vol6_split"
SAMPLE_RATE = 16000
CHUNK_DURATION = 4.0  # seconds per split (recommended: 2–6 sec)

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# SPLIT FUNCTION
# =========================
def split_wav(file_path, output_path, chunk_duration, sample_rate):
    waveform, sr = torchaudio.load(file_path)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate

    num_samples = waveform.shape[1]
    chunk_size = int(chunk_duration * sr)
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_samples)
        chunk = waveform[:, start:end]

        # Skip too short clips (<1 sec)
        if chunk.shape[1] < sr:  
            continue

        out_file = os.path.join(output_path, f"chunk_{i:03d}.wav")
        torchaudio.save(out_file, chunk, sr)

    print(f"Done! Split into {num_chunks} chunks (saved at {output_path})")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    split_wav(INPUT_FILE, OUTPUT_PATH, CHUNK_DURATION, SAMPLE_RATE)
