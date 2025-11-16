import sys
import os

# Add src directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
import numpy as np
import io
import requests
import soundfile as sf 

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
# Load model directly
from transformers import AutoProcessor, AutoModelForTextToSpectrogram

# Import extract_text for handling various input types
from preprocessing.extract_text import extract_text
from preprocessing.cleaning_text.text_normalization import TextNormalizer


app = FastAPI()

# Load models once at startup
processor = AutoProcessor.from_pretrained("Klein2303/speecht5_finetuned_voxpopuli_en")
model = AutoModelForTextToSpectrogram.from_pretrained("Klein2303/speecht5_finetuned_voxpopuli_en")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Default speaker embedding (no voice cloning)
speaker_embeddings = torch.zeros((1, 512))

# Initialize text normalizer
normalizer = TextNormalizer(language="en")


def generate_audio(text: str, voice_array=None):
    """
    Generate numpy audio array from text using SpeechT5.
    Optionally takes a voice embedding (voice_array).
    Automatically chunks text if it's too long.
    """
    # Tokenize to check length
    inputs = processor(text=text, return_tensors="pt")
    
    # If text is too long, chunk it
    max_length = 600  # SpeechT5 max input length
    if inputs["input_ids"].shape[1] > max_length:
        # Split into sentences or chunks
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        audio_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            test_inputs = processor(text=test_chunk, return_tensors="pt")
            
            if test_inputs["input_ids"].shape[1] > max_length:
                # Process current chunk if it exists
                if current_chunk:
                    audio_chunks.append(_generate_audio_single(current_chunk, voice_array))
                    current_chunk = ""
                
                # If the sentence itself is too long, split it by words
                sentence_inputs = processor(text=sentence, return_tensors="pt")
                if sentence_inputs["input_ids"].shape[1] > max_length:
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        test_word_chunk = word_chunk + " " + word if word_chunk else word
                        test_word_inputs = processor(text=test_word_chunk, return_tensors="pt")
                        if test_word_inputs["input_ids"].shape[1] > max_length:
                            if word_chunk:
                                audio_chunks.append(_generate_audio_single(word_chunk, voice_array))
                            word_chunk = word
                        else:
                            word_chunk = test_word_chunk
                    if word_chunk:
                        audio_chunks.append(_generate_audio_single(word_chunk, voice_array))
                else:
                    current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        # Process remaining chunk
        if current_chunk:
            audio_chunks.append(_generate_audio_single(current_chunk, voice_array))
        
        # Concatenate all chunks
        return np.concatenate(audio_chunks) if audio_chunks else np.array([])
    else:
        return _generate_audio_single(text, voice_array)


def _generate_audio_single(text: str, voice_array=None):
    """Generate audio for a single chunk of text."""
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
    try:
        # 1️⃣ Determine text source
        if file is not None:
            content = (await file.read()).decode("utf-8")
        elif text:
            content = text
        elif link:
            try:
                # Use extract_text to properly handle YouTube, PDFs, websites, etc.
                content = extract_text(link)
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"Error extracting from link: {error_detail}")
                return JSONResponse({"error": f"Failed to extract content from link: {str(e)}"}, status_code=400)
        else:
            return JSONResponse({"error": "Please provide text, file, or link."}, status_code=400)

        # Normalize the extracted text
        try:
            content = normalizer.normalize(content)
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"Error normalizing text: {error_detail}")
            return JSONResponse({"error": f"Failed to normalize text: {str(e)}"}, status_code=400)

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
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Unhandled error in tts_endpoint: {error_detail}")
        return JSONResponse({"error": f"Internal server error: {str(e)}"}, status_code=500)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")

