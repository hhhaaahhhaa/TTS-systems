import os
import numpy as np
import torch
import json
from scipy.io import wavfile
import matplotlib.pylab as plt

from dlhlp_lib.audio import AUDIO_CONFIG, STFT
from dlhlp_lib.audio.vocoders import get_vocoder, BaseVocoder
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
	mel_outputs, mel_outputs_postnet, alignments = output
	plot_data((to_arr(mel_outputs[0]),
				to_arr(mel_outputs_postnet[0]),
				to_arr(alignments[0]).T))
	plt.savefig(pth)


def build_tacotron2(ckpt_path: str):
    system = get_system("tacotron2")
    model = system.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()

    return model


def build_vocoder(system_type: str):
    vocoder_cls = get_vocoder(system_type)
    if system_type == "GriffinLim":
        vocoder = vocoder_cls(STFT)
    else:
        vocoder = vocoder_cls().cuda()
    return vocoder


def inference(system, text, spk, lang_id, img_path, mel_path):
    text = torch.from_numpy(text).unsqueeze(0).long()
    spk = torch.from_numpy(spk).long()

    with torch.no_grad():
        output = system.model.inference(text, spk, lang_id)
    
    plot(output, pth=img_path)
    mel_outputs, mel_outputs_postnet, _ = output
    np.save(mel_path, to_arr(mel_outputs_postnet[0]))


def mel2wav(vocoder: BaseVocoder, mel_path, wav_path):
    mel = torch.from_numpy(np.load(mel_path)).float().cuda()
    with torch.no_grad():
        wav = vocoder.infer(mel.unsqueeze(0))[0]
    wavfile.write(wav_path, AUDIO_CONFIG["audio"]["sampling_rate"], wav)


if __name__ == "__main__":
    # ==================parameters==================
    ckpt_path = "output/debug-final6/ckpt/epoch=3-step=10000.ckpt"
    data_config = "data_config/LJSpeech-1.1"
    input = "Deep learning is fun."
    spk = "LJSpeech"  # "LJSpeech", "103"...
    
    output_img_path = "_temp/test.png"
    output_mel_path = "_temp/test.npy"
    output_wav_path = "_temp/test.wav"
    vocoder = "MelGAN"  # 'MelGAN' or 'HifiGAN'
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
    vocoder = build_vocoder(vocoder)
    system = build_tacotron2(ckpt_path).cuda()
    system.eval()

    # parser input to model's input format
    text = text_to_sequence(input, data_config["text_cleaners"], data_config["lang_id"])
    with open(Define.DATAPARSERS[data_config["name"]].speakers_path, 'r') as f:
        speakers = json.load(f)
    spk = np.array([speakers.index(spk)])

    inference(system, text, spk, data_config["lang_id"], output_img_path, output_mel_path)
    mel2wav(vocoder, output_mel_path, output_wav_path)
