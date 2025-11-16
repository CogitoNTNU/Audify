import gradio as gr
import requests
import sys
import os
import tempfile
import soundfile as sf
import numpy as np
import re

sys.path.append(os.path.abspath("src/voice_cloning"))
from voice_cloning.voice_clone import voice_cloning
from preprocessing.cleaning_text.text_normalization import TextNormalizer

API_URL = "http://127.0.0.1:8000/tts/"

# Initialize text normalizer
normalizer = TextNormalizer(language="en")

def chunk_text(text, max_len=50):
    """
    Splits text into natural chunks based on:
    - Sentence endings (.!?)
    - Commas
    - Character length
    """
    text = text.strip()
    # Split by sentence and commas, but keep the punctuation
    parts = re.split(r'([.,!?])', text)
    chunks = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        current += part + " "
        if len(current) >= max_len or part in [".", "!", "?", ","]:
            chunks.append(current.strip())
            current = ""

    if current:
        chunks.append(current.strip())

    return chunks


def call_tts(text, file, link, save_audio):
    files = {}
    data = {"save": str(save_audio).lower()}

    # Priority: file > text > link
    if file is not None:
        with open(file.name, "r", encoding="utf-8") as f:
            text = f.read()
    elif link:
        # When link is provided, let the API handle extraction
        # Send link to API and get the text processed there
        data["link"] = link
        text = "link_provided"  # Placeholder to pass the check
    
    if not text or text == "":
        return "Please provide text, file, or link.", None

    # ---- If we have actual text (not a link), normalize it ----
    if text != "link_provided":
        text = normalizer.normalize(text)
        print(f"📝 Normalized text: {text[:200]}...")  # Show first 200 chars

        # ---- Chunking ----
        chunks = chunk_text(text)
        print(f"🔹 Chunked into {len(chunks)} parts: {chunks}")

        # ---- Generate TTS for each chunk ----
        temp_wavs = []
        for i, chunk in enumerate(chunks):
            response = requests.post(API_URL, data={"text": chunk, "save": "false"})
            if response.status_code == 200 and "audio/wav" in response.headers.get("content-type", ""):
                with tempfile.NamedTemporaryFile(suffix=f"_{i}.wav", delete=False) as tmp:
                    tmp.write(response.content)
                    temp_wavs.append(tmp.name)
            else:
                return f"❌ Error on chunk {i+1}: {response.text}", None
        num_chunks = len(chunks)
    else:
        # ---- For links, let API handle everything ----
        response = requests.post(API_URL, data=data)
        if response.status_code == 200 and "audio/wav" in response.headers.get("content-type", ""):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(response.content)
                temp_wavs = [tmp.name]
            num_chunks = 1  # API handles chunking internally
        else:
            return f"❌ Error: {response.text}", None

    # ---- Combine all chunks with pauses ----
    combined_audio = []
    samplerate = None
    pause_duration = 0.3  # seconds of silence between chunks

    for path in temp_wavs:
        data, sr = sf.read(path)
        if samplerate is None:
            samplerate = sr
        combined_audio.append(data)

        # Add short silence between chunks
        pause = np.zeros(int(sr * pause_duration))
        combined_audio.append(pause)

    final_audio = np.concatenate(combined_audio)
    final_path = "final_combined.wav"
    sf.write(final_path, final_audio, samplerate)

    # Cleanup temp files
    for path in temp_wavs:
        os.remove(path)

    # Convert to int16 to avoid Gradio warning
    final_audio = (final_audio * 32767).astype(np.int16)

    return f"✅ Speech generated successfully! Combined {num_chunks} chunks.", (samplerate, final_audio)


def call_clone(voice_file, original_audio, chunk_size):
    if isinstance(original_audio, tuple):
        sr, audio_np = original_audio
    else:
        sr = 16000
        audio_np = original_audio

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_np, sr)
        tmp_path = tmp.name

    voice_cloning(voice_file, tmp_path, target_sec=chunk_size)
    final_path = "final_product.wav"

    cloned_audio, cloned_sr = sf.read(final_path, dtype="int16")
    return f"✅ Cloning done! Saved as {final_path}", (cloned_sr, cloned_audio)


# ---- Gradio UI ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Audify TTS Generator  
        Convert text to speech using SpeechT5.  
        Upload a `.txt` file, type directly, or paste a link.  
        Automatically chunks text into natural phrases for better clarity.  
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 📝 Input")
            text_input = gr.Textbox(label="Text", placeholder="Type or paste text...", lines=4)
            file_input = gr.File(label="Upload .txt File (optional)", file_types=[".txt"])
            link_input = gr.Textbox(label="Link (optional)")
            save_checkbox = gr.Checkbox(label="💾 Save audio to server", value=False)

            gr.Markdown("## 🎤 Voice Cloning")
            voice_input = gr.File(label="Voice (.wav) (optional)", file_types=[".wav"])
            chunk_input = gr.Number(label="Chunk Size (optional)", value=15, precision=0)

        with gr.Column(scale=1):
            gr.Markdown("## 🎧 Output")
            status_output = gr.Textbox(label="Status", interactive=False)
            audio_output = gr.Audio(label="Generated Audio", type="numpy")
            submit_btn = gr.Button("🔊 Generate Speech")

            gr.Markdown("## 🎧 Cloned Output")
            clone_status = gr.Textbox(label="Status", interactive=False)
            clone_output = gr.Audio(label="Cloned Audio", type="numpy")
            clone_btn = gr.Button("🧬 Clone Voice")

    submit_btn.click(
        fn=call_tts,
        inputs=[text_input, file_input, link_input, save_checkbox],
        outputs=[status_output, audio_output]
    )

    clone_btn.click(
        fn=call_clone,
        inputs=[voice_input, audio_output, chunk_input],
        outputs=[clone_status, clone_output]
    )

demo.launch()
