import os
import numpy as np
import torch
import json
from scipy.io import wavfile

from dlhlp_lib.audio import AUDIO_CONFIG, STFT
from dlhlp_lib.audio.vocoders import get_vocoder, BaseVocoder

import Define
from global_setup import setup_data
from tts.systems import get_system
from text import text_to_sequence


def build_fastspeech2(ckpt_path: str):
    system = get_system("fastspeech2")
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


def inference(system, text, spk, lang_id, mel_path, p_control=1.0, e_control=1.0, d_control=1.0):
    text_lens = np.array([len(text)])

    text = torch.from_numpy(text).unsqueeze(0).long()
    text_lens = torch.from_numpy(text_lens)
    max_text_len = max(text_lens)
    spk = torch.from_numpy(spk).long()

    with torch.no_grad():
        emb_texts = system.embedding_layer(text.cuda(), lang_id)
        mel = system.model(
            spk.cuda(), emb_texts, text_lens.cuda(), max_text_len.cuda(), 
            p_control=p_control, e_control=e_control, d_control=d_control
        )[1]
        mel = mel[0].detach().cpu().numpy()
     
    with open(mel_path, 'wb') as f:
        np.save(f, mel)


def mel2wav(vocoder: BaseVocoder, mel_path, wav_path):
    mel = torch.from_numpy(np.load(mel_path).T).float().cuda()
    with torch.no_grad():
        wav = vocoder.infer(mel.unsqueeze(0))[0]
    wavfile.write(wav_path, AUDIO_CONFIG["audio"]["sampling_rate"], wav)


if __name__ == "__main__":
    # parameters
    ckpt_path = "output/debug-final6/ckpt/epoch=3-step=10000.ckpt"
    data_config = "data_config/LJSpeech-1.1"
    input = "{T IH1 T IH1 EH1 S IH1 Z F UH1 N}"
    spk = "LJSpeech"
    control = {  # Control FastSpeech2
        "p_control": 1.0,
        "e_control": 1.0,
        "d_control": 1.0,
    }
    
    os.makedirs("_temp", exist_ok=True)
    output_mel_path = "_temp/test.npy"
    output_wav_path = "_temp/test.wav"

    vocoder = "HifiGAN"  # 'MelGAN' or 'HifiGAN'

    # inference
    from Objects.config import DataConfigReader
    reader = DataConfigReader()
    data_config = reader.read(data_config)
    setup_data([data_config])  # data config is used to reconstruct pitch and energy normalization statisitics.

    # build model
    vocoder = build_vocoder(vocoder)
    system = build_fastspeech2(ckpt_path).cuda()
    system.eval()

    # parser input to model's input format
    text = np.array(text_to_sequence(input, data_config["text_cleaners"], data_config["lang_id"]))
    with open(Define.DATAPARSERS[data_config["name"]].speakers_path, 'r') as f:
        speakers = json.load(f)
    spk = np.array([speakers.index(spk)])

    inference(system, text, spk, data_config["lang_id"], output_mel_path, **control)
    mel2wav(vocoder, output_mel_path, output_wav_path)
