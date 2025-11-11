import gradio as gr
import requests
from preprocessing.extract_text import extract_text

API_URL = "http://127.0.0.1:8000/tts/"

def call_tts(text, file, link, voice_file, chunk_size, save_audio):
    """
    Calls the FastAPI backend /tts/ endpoint.
    Handles text extraction from file or link, optional voice cloning, and chunk size.
    """
    files = {}
    data = {"save": str(save_audio).lower()}

    # 1️⃣ Extract text from file or link
    try:
        if file is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            extracted_text = extract_text(tmp_path)
        elif link:
            extracted_text = extract_text(link)
        elif text:
            extracted_text = text
        else:
            return "Please provide text, file, or link.", None
    except Exception as e:
        return f"❌ Failed to extract text: {e}", None

    # 2️⃣ Optional voice cloning
    if voice_file is not None:
        files["voice"] = (voice_file.name, voice_file.read(), "audio/wav")

    # 3️⃣ Optional chunk size
    if chunk_size is not None and chunk_size > 0:
        data["chunk_size"] = str(int(chunk_size))

    # 4️⃣ Send request to backend
    response = requests.post(API_URL, data={**data, "text": extracted_text}, files=files if files else None)

    if response.status_code == 200:
        content_type = response.headers.get("content-type", "")
        if "audio/wav" in content_type:
            return "✅ Speech generated successfully!", response.content
        else:
            # Backend saved audio to server
            return response.json().get("message", "✅ Audio saved to server."), None
    else:
        return f"❌ Error {response.status_code}: {response.text}", None


# -------------------- Gradio UI --------------------
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
            gr.Markdown("## 📝 What should the audio say?")
            text_input = gr.Textbox(label="Text Input", placeholder="Type something...", lines=4)
            file_input = gr.File(label="Upload .txt File (optional)", file_types=[".txt"])
            link_input = gr.Textbox(label="Link Input (optional)", placeholder="Paste a URL here...")
            save_checkbox = gr.Checkbox(label="💾 Save audio to server", value=False)

            gr.Markdown("## 🎤 Voice Cloning Options")
            voice_input = gr.File(label="Upload Voice (.wav) for Cloning (optional)", file_types=[".wav"])
            chunk_input = gr.Number(label="Chunk Size (optional)", value=0, precision=0)

        with gr.Column(scale=1):
            gr.Markdown("## 🎧 Output")
            status_output = gr.Textbox(label="Status", interactive=False)
            audio_output = gr.Audio(label="Generated Audio", type="numpy")
            submit_btn = gr.Button("🔊 Generate Speech")

    submit_btn.click(
        fn=call_tts,
        inputs=[text_input, file_input, link_input, voice_input, chunk_input, save_checkbox],
        outputs=[status_output, audio_output]
    )

demo.launch()
