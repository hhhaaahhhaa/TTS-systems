import os
import numpy as np
import torch
import json
from scipy.io import wavfile
import matplotlib.pylab as plt

from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.vocoders import get_vocoder, BaseVocoder
from dlhlp_lib.utils import to_arr

import Define
from global_setup import setup_data
from tts.systems import get_system
from text import text_to_sequence


def plot_data(data, figsize = (16, 4)):
	fig, axes = plt.subplots(1, len(data), figsize = figsize)
	for i in range(len(data)):
		axes[i].imshow(data[i], aspect = 'auto', origin = 'lower')


def plot(output, pth):
	os.makedirs(os.path.dirname(pth), exist_ok=True)
	mel_outputs, mel_outputs_postnet, _, alignments = output
	plot_data((to_arr(mel_outputs[0]),
				to_arr(mel_outputs_postnet[0]),
				to_arr(alignments[0]).T))
	plt.savefig(pth)


def build_tacotron2(ckpt_path: str, data_configs):
    system = get_system("tacotron2")
    model = system.load_from_checkpoint(data_configs=data_configs, checkpoint_path=ckpt_path)
    model.eval()

    return model


def inference(system, text, spk, lang_id, img_path, mel_path):
    text = torch.from_numpy(text).unsqueeze(0).long().cuda()
    spk = torch.from_numpy(spk).long().cuda()

    with torch.no_grad():
        output = system.model.inference(text, spk, [lang_id])
    
    plot(output, pth=img_path)
    mel_outputs, mel_outputs_postnet, _, _ = output
    np.save(mel_path, to_arr(mel_outputs_postnet[0]))


def mel2wav(vocoder: BaseVocoder, mel_path, wav_path):
    mel = torch.from_numpy(np.load(mel_path)).float().cuda()
    with torch.no_grad():
        wav = vocoder.infer(mel.unsqueeze(0))[0]
    wavfile.write(wav_path, AUDIO_CONFIG["audio"]["sampling_rate"], wav)


if __name__ == "__main__":
    # ==================parameters==================
    ckpt_path = ""
    data_config = "data_config/LibriTTS"
    input = "Deep learning is fun."
    spk = "103"  # "LJSpeech", "103"...
    
    output_img_path = "_temp/test.png"
    output_mel_path = "_temp/test.npy"
    output_wav_path = "_temp/test.wav"
    vocoder = "HifiGAN"
    # ==================parameters==================
    
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_mel_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)

    # inference
    from Objects.config import DataConfigReader
    reader = DataConfigReader()
    data_config = reader.read(data_config)
    setup_data([data_config])

    # build model
    vocoder = get_vocoder(vocoder)().cuda()
    system = build_tacotron2(ckpt_path, [data_config]).cuda()
    system.eval()

    # parser input to model's input format
    text = np.array(text_to_sequence(input, data_config["text_cleaners"], data_config["lang_id"]))

    # If you want to use apply g2p and use phoneme as input, use functions from fastspeech2_inference.py
    # from fastspeech2_inference import preprocess_english, preprocess_mandarin
    # if data_config["lang_id"] == "en":
    #     text = preprocess_english(input)
    # elif data_config["lang_id"] == "zh":
    #     text = preprocess_mandarin(input)
    # else:
    #     raise NotImplementedError

    with open(Define.DATAPARSERS[data_config["name"]].speakers_path, 'r') as f:
        speakers = json.load(f)
    spk = np.array([speakers.index(spk)])

    inference(system, text, spk, data_config["lang_id"], output_img_path, output_mel_path)
    mel2wav(vocoder, output_mel_path, output_wav_path)
