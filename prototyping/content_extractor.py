#ALTERNATIVE TO PREPROCESS.PY (?)

from transformers import Wav2Vec2FeatureExtractor, WavLMModel
import torch
import torchaudio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract(wav_path):
    """
    Input: wav file path
    

    Output: Latent content
    """

    # Load pretrained WavLM
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE)
    model.eval()

    # Load waveform
    y, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        y = resampler(y)
    y = y.squeeze(0)

    # Convert to input values
    inputs = feature_extractor(y.numpy(), sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_values)
    latent_content = outputs.last_hidden_state.squeeze(0)  # (Time, Hidden_dim)

    return latent_content 
