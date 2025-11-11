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
import soundfile as sf

app = FastAPI()

# Load models once at startup
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Default speaker embedding (no voice cloning)
speaker_embeddings = torch.zeros((1, 512))


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


def tts(text: str, voice_array=None, chunk_size: Optional[int] = None):
    """
    Generate audio from text with optional chunking and voice cloning.
    """
    if chunk_size and chunk_size > 0:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        audios = [generate_audio(chunk, voice_array) for chunk in chunks]
        return np.concatenate(audios)
    else:
        return generate_audio(text, voice_array)


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

    # 2️⃣ Handle optional voice cloning (not implemented fully yet)
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

