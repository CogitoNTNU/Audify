from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import os
import uuid
import torch
import numpy as np
import soundfile as sf  # For saving audio
from src.tts_service import tts


from transformers import SpeechT5HifiGan, SpeechT5ForTextToSpeech, SpeechT5Processor

app = FastAPI()

@app.post("/tts/")      
async def text_to_speech(
    text: str = Form(None),
    file: UploadFile = File(None),
    save: bool = Form(False)
):
    # Get text from file or form
    if file:
        contents = await file.read()
        text = contents.decode("utf-8")

    if not text:
        return {"error": "No text provided"}

    # Generate speech using SpeechT5
    audio_array = tts(text)
    filename = f"{uuid.uuid4()}.wav"
    sf.write(filename, audio_array, samplerate=16000)

    # Optional save
    if save:
        saved_path = os.path.join("saved_audio", filename)
        os.makedirs("saved_audio", exist_ok=True)
        os.rename(filename, saved_path)
        return {"message": "Audio saved", "path": saved_path}

    # Return audio file
    return FileResponse(filename, media_type="audio/wav", filename=filename)
