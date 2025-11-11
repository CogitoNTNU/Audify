
#CPU NOT WORKING (NOT ENOUGH RAM, I think), USE GPU INSTEAD
#also very very slow

##### CPU TESTING CODE #####
# import torch
# torch.set_num_threads(4) 

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



    # very large huber pair
    # hubert_model_name = 'V1_ubert_base_ls960_23.pth'

    # 14 epochs
    # hubert_model_name = 'hubert_base_ls960.pth'

    # 4 epochs, small huber pair
    # hubert_model_name = 'hubert_base_ls960_14.pth'

import model_downloader

orig_func = model_downloader.get_hubert_manager_and_model

#Injector
def patched_get_hubert_manager_and_model(*args, **kwargs):
    kwargs.setdefault("hubert_model_name", "hubert_base_ls960.pth")
    return orig_func(*args, **kwargs)

model_downloader.get_hubert_manager_and_model = patched_get_hubert_manager_and_model # Monkey patch it



import utils
utils.get_cpu_or_gpu = lambda: "cuda" #"cpu" #for manually setting device (if you have GPU, defaults to cuda)

###############################

from voice2embedding import voice2embedding
from voice2voice import voice2voice


print("EMBEDDING")
# speaker embedding generation
embedding = voice2embedding(audio_file="input1.wav", voice_name="test").save_to_speaker_lib() #vocie to keep
print("EMBEDDING DONE")

print("CLONING")
# voice2voice synthesis
cloned_audio, sample_rate = voice2voice(audio_file="input2.wav", voice_name="test") #audio to be replaced
print("CLONING DONE")


print("SAVING OUTPUT")
from media_toolkit import AudioFile
audio = AudioFile().from_np_array(cloned_audio, sample_rate=sample_rate)
audio.save("v_output.wav")
print("SAVED OUTPUT")


#Optimal: 1 min input2
#Solution: Split up longer audios into chunks of ~1 min, process each chunk, then combine the outputs

#Expected inputs
#Sample rate: 16 kHz
#Mono audio


