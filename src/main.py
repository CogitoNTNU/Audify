from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
import numpy as np
import io
import requests
import os
import soundfile as sf

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch

app = FastAPI()

# Load models once at startup
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Default speaker embedding (no voice cloning)
speaker_embeddings = torch.zeros((1, 512))

# Safe maximum token length for SpeechT5
MAX_TOKENS = 600  # adjust if necessary


def generate_audio(text: str, voice_array=None):
    """
    Generate numpy audio array from text using SpeechT5.
    Optionally takes a voice embedding (voice_array).
    """
    inputs = processor(text=text, return_tensors="pt")

    speaker = speaker_embeddings
    if voice_array is not None:
        try:
            speaker = torch.tensor(voice_array).unsqueeze(0).float()
        except Exception:
            pass  # fallback to default speaker

    speech = model.generate_speech(inputs["input_ids"], speaker, vocoder=vocoder)
    return speech.numpy()


def chunk_text(text: str, max_tokens: int = MAX_TOKENS):
    """
    Split text into smaller chunks to avoid exceeding model's max token length.
    This uses a naive character-based approximation.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_tokens, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

def tts(text: str, voice_array=None, chunk_size: Optional[int] = None):
    """
    Generate audio from text with optional chunking and voice cloning.
    Splits long text into chunks to avoid model max token limits.
    """
    # If a chunk_size is provided, use it; otherwise, default to MAX_TOKENS
    step = chunk_size or MAX_TOKENS

    # Split text into chunks (naive character-based, could also use word-based)
    chunks = [text[i:i+step] for i in range(0, len(text), step)]

    audio_segments = []
    for chunk in chunks:
        audio_segments.append(generate_audio(chunk, voice_array))

    # Concatenate all audio segments
    return np.concatenate(audio_segments)



@app.post("/tts/")
async def tts_endpoint(
    text: Optional[str] = Form(None),
    link: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    save: Optional[str] = Form("false"),
    file: Optional[UploadFile] = File(None),
    voice: Optional[UploadFile] = File(None)
):
    """
    Endpoint for Text-to-Speech generation.
    Supports:
    - text input
    - text file input
    - link input (fetch text from URL)
    - optional voice cloning (.wav)
    - optional chunk size
    - optional saving to server
    """

    # 1️⃣ Determine text source
    if file is not None:
        content = (await file.read()).decode("utf-8")
    elif text:
        content = text
    elif link:
        try:
            resp = requests.get(link)
            content = resp.text
        except Exception:
            return JSONResponse({"error": "Failed to fetch link content"}, status_code=400)
    else:
        return JSONResponse({"error": "Please provide text, file, or link."}, status_code=400)

    # 2️⃣ Handle optional voice cloning
    voice_array = None
    if voice is not None:
        data, sr = sf.read(io.BytesIO(await voice.read()))
        voice_array = data  # placeholder for speaker embedding extraction

    # 3️⃣ Convert chunk_size
    if chunk_size is not None:
        try:
            chunk_size = int(chunk_size)
        except ValueError:
            chunk_size = None

    # 4️⃣ Generate speech
    audio_array = tts(content, voice_array, chunk_size)

    # 5️⃣ Save if requested
    if save.lower() == "true":
        os.makedirs("outputs", exist_ok=True)
        filename = f"outputs/tts_output.wav"
        sf.write(filename, audio_array, 16000)
        return JSONResponse({"message": f"Audio saved to {filename}"})

    # 6️⃣ Return as stream
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, 16000, format="WAV")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")
