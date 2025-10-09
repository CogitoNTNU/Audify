import torch
import yaml
import json
from pathlib import Path
from hifi_gan_models import Generator  # from HiFi-GAN repo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_hifigan_model(checkpoint_path, config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    mel_channels = config["num_mels"]
    generator = Generator(mel_channels=mel_channels)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    generator.load_state_dict(state_dict)
    generator.to(DEVICE).eval()
    return generator

def mel_to_waveform_hifigan(mel, generator):
    """
    mel: np.ndarray (T, n_mels)
    """
    import torch
    mel_tensor = torch.tensor(mel.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, n_mels, T)
    with torch.no_grad():
        audio = generator(mel_tensor)
    return audio.squeeze(0).cpu()
