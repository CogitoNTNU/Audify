import os
import re
import shutil
import torchaudio
import torch
from .voice2embedding import voice2embedding
from .voice2voice import voice2voice
from media_toolkit import AudioFile
from pydub import AudioSegment  # only used for input2 splitting


def voice_cloning(input1_path="input1.wav", input2_path="input2.wav", target_sec=15):
    # -------------------------
    # Monkey patch HuBERT loader
    # -------------------------
    from . import model_downloader

    orig_func = model_downloader.get_hubert_manager_and_model

    ##### Choose between these models below #####

    # very large hubert pair
    # 'V1_ubert_base_ls960_23.pth'

    # 14 epochs
    # 'hubert_base_ls960.pth'

    # 4 epochs, small hubert pair
    # 'hubert_base_ls960_14.pth'

    # Injector: add default hubert_model_name argument
    def patched_get_hubert_manager_and_model(*args, **kwargs):
        kwargs.setdefault("hubert_model_name", "hubert_base_ls960.pth")
        return orig_func(*args, **kwargs)

    model_downloader.get_hubert_manager_and_model = patched_get_hubert_manager_and_model  # Monkey patch it

    ##########################################################

    # -------------------------
    # Ensure folders exist
    # -------------------------
    os.makedirs("input2_files", exist_ok=True)
    os.makedirs("input2_outputs", exist_ok=True)

    # -------------------------
    # Step 0: Convert input2 to mono 16kHz
    # -------------------------
    def convert_to_16k_mono(input_path, output_path):
        audio, sr = torchaudio.load(input_path)
        if audio.shape[0] > 1:  # stereo to mono
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = transform(audio)
        torchaudio.save(output_path, audio, 16000)
        return output_path

    converted_path = "input2_converted.wav"
    convert_to_16k_mono(input2_path, converted_path)

    # -------------------------
    # Step 1: Split input2 into ~x second chunks
    # -------------------------
    audio2 = AudioSegment.from_wav(converted_path)
    duration_ms = len(audio2)
    chunk_length_ms = target_sec * 1000

    if duration_ms > chunk_length_ms:
        print(f"Audio is longer than {target_sec} seconds — splitting into chunks...")
        for i, start in enumerate(range(0, duration_ms, chunk_length_ms)):
            chunk = audio2[start:start + chunk_length_ms]
            chunk.export(f"input2_files/chunk_{i+1}.wav", format="wav")
    else:
        audio2.export("input2_files/chunk_1.wav", format="wav")

    # -------------------------
    # Step 2: Run cloning for each chunk, padding input1 per chunk
    # -------------------------
    print("CLONING")

    def numeric_sort_key(filename):
        """Extract numeric part from filename for proper sorting."""
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else -1

    def pad_input1_to_length(input1_path, chunk_length_sec):
        """Load input1 and loop/pad it to match chunk length"""
        audio_tensor, sr = torchaudio.load(input1_path)
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
        duration_sec = audio_tensor.shape[1] / sr
        if duration_sec < chunk_length_sec:
            repeats = int(chunk_length_sec // duration_sec) + 1
            audio_tensor = audio_tensor.repeat(1, repeats)
            audio_tensor = audio_tensor[:, :int(chunk_length_sec * sr)]
        return audio_tensor, sr

    output_files = []
    for file_name in sorted(os.listdir("input2_files"), key=numeric_sort_key):
        if not file_name.endswith(".wav"):
            continue

        input_file = os.path.join("input2_files", file_name)
        # Load chunk to get its duration
        chunk_audio = AudioSegment.from_wav(input_file)
        chunk_length_sec = len(chunk_audio) / 1000

        # Step 2a: Pad/loop input1 to match chunk length
        input1_tensor, input1_sr = pad_input1_to_length(input1_path, chunk_length_sec)
        input1_padded_path = f"input1_padded_{file_name}"
        torchaudio.save(input1_padded_path, input1_tensor, input1_sr)

        # -------------------------
        # Step 3: Generate speaker embedding
        # -------------------------
        print(f"EMBEDDING for {file_name}")
        embedding = voice2embedding(audio_file=input1_padded_path, voice_name="test").save_to_speaker_lib()
        print(f"EMBEDDING DONE for {file_name}")

        # -------------------------
        # Step 4: Clone chunk
        # -------------------------
        cloned_audio, sample_rate = voice2voice(audio_file=input_file, voice_name="test")  # audio to be replaced
        output_file = os.path.join("input2_outputs", f"output_{file_name}")
        audio_out = AudioFile().from_np_array(cloned_audio, sample_rate=sample_rate)
        audio_out.save(output_file)
        output_files.append(output_file)
        print(f"Processed {file_name} → {output_file}")

        # -------------------------
        # Step 4a: Delete temporary padded input1 file
        # -------------------------
        os.remove(input1_padded_path)

    print("CLONING DONE")

    # -------------------------
    # Step 5: Combine all outputs into final product
    # -------------------------
    print("COMBINING OUTPUTS")

    combined = AudioSegment.empty()
    for out_file in sorted(output_files, key=numeric_sort_key):
        combined += AudioSegment.from_wav(out_file)

    combined.export("final_product.wav", format="wav")
    print("Final product saved as final_product.wav ✅")

    # -------------------------
    # Step 6: Cleanup
    # -------------------------
    print("Cleaning up temporary files...")
    shutil.rmtree("input2_files", ignore_errors=True)
    shutil.rmtree("input2_outputs", ignore_errors=True)
    print("Temporary folders deleted ✅")

    # IMPORTANT! input1 needs to be equal length to input2 chunks for best results #FIXED
    # Perhaps target_sec needs to be around 10 sec? (For some reason male2female goes very deep after ~15,17,20 seconds per chunk)
    # Possible fix: increase pitch (and standardize pitch?) of input2 before processing???
    # In front end: Sliders for chunk length and pitch

    # Disregard Triton errors
    # 30-60 second chunks: 1.5-5 min per chunk on GPU
    # 15 second chunks: 20-30 seconds
