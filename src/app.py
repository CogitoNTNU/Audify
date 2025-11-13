import gradio as gr
import requests
import os

API_URL = "http://127.0.0.1:8000/tts/"

def call_tts(text, file, save_audio):
    files = {}
    data = {"save": str(save_audio).lower()}

    if file is not None:
        files["file"] = (file.name, file.read(), "text/plain")
    elif text:
        data["text"] = text
    else:
        return "Please provide either text or a file.", None

    response = requests.post(API_URL, data=data, files=files if files else None)

    if response.status_code == 200:
        if "audio/wav" in response.headers.get("content-type", ""):
            return "✅ Speech generated successfully!", response.content
        else:
            return response.json().get("message", "✅ Audio saved to server."), None
    else:
        return f"❌ Error {response.status_code}: {response.text}", None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Audify TTS Generator
        Convert text to speech using SpeechT5 and FastAPI.  
        Upload a `.txt` file or type directly. Choose whether to save the audio to the server.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="📝 Text Input", placeholder="Type something...", lines=4)
            file_input = gr.File(label="📄 Upload .txt File (optional)", file_types=[".txt"])
            save_checkbox = gr.Checkbox(label="💾 Save audio to server", value=False)
            submit_btn = gr.Button("🔊 Generate Speech")

        with gr.Column(scale=1):
            status_output = gr.Textbox(label="Status", interactive=False)
            audio_output = gr.Audio(label="🎧 Generated Audio", type="numpy")

    submit_btn.click(
        fn=call_tts,
        inputs=[text_input, file_input, save_checkbox],
        outputs=[status_output, audio_output]
    )

demo.launch(
    server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
    server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
    share=False
)
