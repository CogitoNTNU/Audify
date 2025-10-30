from transformers import SpeechT5HifiGan, SpeechT5ForTextToSpeech, SpeechT5Processor
import torch
import numpy as np

# Load model directly
from transformers import AutoProcessor, AutoModelForTextToSpectrogram

processor = AutoProcessor.from_pretrained("Klein2303/speecht5_finetuned_voxpopuli_en")
model = AutoModelForTextToSpectrogram.from_pretrained("Klein2303/speecht5_finetuned_voxpopuli_en")


# model = SpeechT5ForTextToSpeech.from_pretrained(
#   "Klein2303/speecht5_finetuned_voxpopuli_en"
# )
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
# processor = AutoProcessor.from_pretrained("Klein2303/speecht5_finetuned_voxpopuli_en")

def tts(text: str, speaker_embedding = torch.zeros((1, 512))) -> np.ndarray:
    """
    Convert text to speech using a pre-trained SpeechT5 model.
    Args:
        text (str): The input text to be converted to speech.
        speaker_embedding (torch.Tensor): The speaker embedding tensor of shape (1, 512).
    Returns:
        np.ndarray: The generated speech audio as a numpy array.
    """
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
    return speech.numpy()