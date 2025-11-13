import os.path
import shutil
import urllib.request

import huggingface_hub


class HuBERTManager:
    @staticmethod
    def make_sure_hubert_installed(download_url: str = 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt',
                                   model_path: str = 'hubert_base_ls960'):

        model_dir = os.path.dirname(model_path)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        if not os.path.isfile(model_path):
            print('Downloading HuBERT base model')
            urllib.request.urlretrieve(download_url, model_path)
            print('Downloaded HuBERT')
        return model_path


    @staticmethod
    def make_sure_tokenizer_installed(
            local_tokenizer_path: str,
            model: str = 'hubert_base_ls960_23.pth',
            repo: str = 'GitMylo/bark-voice-cloning',
    ):
        """
        Downloads the tokenizer from the huggingface hub if not already downloaded
        :param local_tokenizer_path: where to save the tokenizer locally
        :param model: the huber model determines the tokenizer model
        :param repo: the github repo to download from
        :return:
        """
        install_dir = os.path.dirname(local_tokenizer_path)
        if not os.path.isdir(install_dir):
            os.makedirs(install_dir, exist_ok=True)

        quantifier_name = f"quantifier_{model}"
        if not os.path.isfile(local_tokenizer_path):
            print('Downloading HuBERT custom tokenizer')
            huggingface_hub.hf_hub_download(repo, quantifier_name, local_dir=install_dir, local_dir_use_symlinks=False)
            shutil.move(os.path.join(install_dir, quantifier_name), local_tokenizer_path)
            print('Downloaded tokenizer')

        return local_tokenizer_path
