import gradio as gr
import requests
import sys
import os
import tempfile
import soundfile as sf

sys.path.append(os.path.abspath("src/voice_cloning"))
from voice_cloning.voice_clone import voice_cloning

API_URL = "http://127.0.0.1:8000/tts/"

def call_tts(text, file, link, save_audio):
    files = {}
    data = {"save": str(save_audio).lower()}

    # Priority: file > text > link
    if file is not None:
        files["file"] = (file.name, file.read(), "text/plain")
    elif text:
        data["text"] = text
    elif link:
        data["link"] = link
    else:
        return "Please provide text, file, or link.", None

    response = requests.post(API_URL, data=data, files=files if files else None)

    if response.status_code == 200:
        if "audio/wav" in response.headers.get("content-type", ""):
            return "✅ Speech generated successfully!", response.content
        else:
            return response.json().get("message", "✅ Audio saved to server."), None
    else:
        return f"❌ Error {response.status_code}: {response.text}", None


import numpy as np
import soundfile as sf

def call_clone(voice_file, original_audio, chunk_size):
    # Save original audio (numpy) to temp WAV
    if isinstance(original_audio, tuple):
        sr, audio_np = original_audio
    else:
        sr = 16000
        audio_np = original_audio

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_np, sr)
        tmp_path = tmp.name

    # Perform voice cloning
    voice_cloning(voice_file, tmp_path, target_sec=chunk_size)

    # The voice_cloning function produces 'final_product.wav'
    final_path = "final_product.wav"

    # Load the WAV to numpy for Gradio audio component
    cloned_audio, cloned_sr = sf.read(final_path, dtype="int16")
    
    # Return status message + numpy array
    return f"✅ Cloning done! Saved as {final_path}", (cloned_sr, cloned_audio)



with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Audify TTS Generator
        Convert text to speech using SpeechT5.  
        Upload a `.txt` file, type directly, or paste a link.  
        Optionally provide a `.wav` file for **voice cloning** and set chunk size.  
        Choose whether to save the audio to the server.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            # Section 1: Input
            gr.Markdown("## 📝 What should the audio say?")
            text_input = gr.Textbox(label="Text Input", placeholder="Type something...", lines=4)
            file_input = gr.File(label="Upload .txt File (optional)", file_types=[".txt"])
            link_input = gr.Textbox(label="Link Input (optional)", placeholder="Paste a URL here...")
            save_checkbox = gr.Checkbox(label="💾 Save audio to server", value=False)

            # Section 2: Voice Cloning
            gr.Markdown("## 🎤 Voice Cloning Options")
            voice_input = gr.File(label="Upload Voice (.wav) for Cloning (optional)", file_types=[".wav"])
            chunk_input = gr.Number(label="Chunk Size (optional)", value=15, precision=0)

        with gr.Column(scale=1):
            # Section 3: Output
            gr.Markdown("## 🎧 Output")
            status_output = gr.Textbox(label="Status", interactive=False)
            audio_output = gr.Audio(label="Generated Audio", type="numpy")
            submit_btn = gr.Button("🔊 Generate Speech")

            # Section 3: Output
            gr.Markdown("## 🎧 Cloned Output")
            clone_status = gr.Textbox(label="Status", interactive=False)
            clone_output = gr.Audio(label="Cloned Audio", type="numpy")
            clone_btn = gr.Button("🔊 Clone Speech")

    submit_btn.click(
        fn=call_tts,
        inputs=[text_input, file_input, link_input, save_checkbox],
        outputs=[status_output, audio_output]
    )

    clone_btn.click(
        fn=call_clone,
        inputs=[voice_input, audio_output, chunk_input],
        outputs=[status_output, clone_output]
    )

demo.launch()
