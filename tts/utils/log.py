import os
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.io import wavfile

from .tool import expand, plot_mel
import Define


def synth_one_sample_with_target(targets, predictions, vocoder, preprocess_config):
    """Synthesize the first sample of the batch given target pitch/duration/energy."""
    basename = targets[0][0]
    src_len         = predictions[8][0].item()
    mel_len         = predictions[9][0].item()
    mel_target      = targets[6][0, :mel_len].detach().transpose(0, 1)
    duration        = targets[11][0, :src_len].detach().cpu().numpy()
    pitch           = targets[9][0, :src_len].detach().cpu().numpy()
    energy          = targets[10][0, :src_len].detach().cpu().numpy()
    mel_prediction  = predictions[1][0, :mel_len].detach().transpose(0, 1)
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = expand(pitch, duration)
    else:
        pitch = targets[9][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = expand(energy, duration)
    else:
        energy = targets[10][0, :mel_len].detach().cpu().numpy()

    # with open(
    #     os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    # ) as f:
    #     stats = json.load(f)
    #     stats = stats["pitch"] + stats["energy"][:2]
    stats = Define.ALLSTATS["global"][:6]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder.mel2wav is not None:
        max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]

        wav_reconstruction = vocoder.infer(mel_target.unsqueeze(0), max_wav_value)[0]
        wav_prediction = vocoder.infer(mel_prediction.unsqueeze(0), max_wav_value)[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def recon_samples(targets, predictions, vocoder, preprocess_config, figure_dir, audio_dir):
    """Reconstruct all samples of the batch."""
    for i in range(len(predictions[0])):
        basename    = targets[0][i]
        src_len     = predictions[8][i].item()
        mel_len     = predictions[9][i].item()
        mel_target  = targets[6][i, :mel_len].detach().transpose(0, 1)
        duration    = targets[11][i, :src_len].detach().cpu().numpy()
        pitch       = targets[9][i, :src_len].detach().cpu().numpy()
        energy      = targets[10][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = expand(pitch, duration)
        else:
            pitch = targets[9][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = expand(energy, duration)
        else:
            energy = targets[10][i, :mel_len].detach().cpu().numpy()

        # with open(
        #     os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        # ) as f:
        #     stats = json.load(f)
        #     stats = stats["pitch"] + stats["energy"][:2]
        stats = Define.ALLSTATS["global"][:6]

        fig = plot_mel(
            [
                (mel_target.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Ground-Truth Spectrogram"],
        )
        plt.savefig(os.path.join(figure_dir, f"{basename}.target.png"))
        plt.close()

    mel_targets = targets[6].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    wav_targets = vocoder.infer(mel_targets, max_wav_value, lengths=lengths)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_targets, targets[0]):
        wavfile.write(os.path.join(audio_dir, f"{basename}.recon.wav"), sampling_rate, wav)


def synth_samples(targets, predictions, vocoder, preprocess_config, figure_dir, audio_dir, name):
    """Synthesize the first sample of the batch."""
    for i in range(len(predictions[0])):
        basename        = targets[0][i]
        src_len         = predictions[8][i].item()
        mel_len         = predictions[9][i].item()
        mel_prediction  = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration        = predictions[5][i, :src_len].detach().cpu().numpy()
        pitch           = predictions[2][i, :src_len].detach().cpu().numpy()
        energy          = predictions[3][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = expand(pitch, duration)
        else:
            pitch = targets[9][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = expand(energy, duration)
        else:
            energy = targets[10][i, :mel_len].detach().cpu().numpy()

        # with open(
        #     os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        # ) as f:
        #     stats = json.load(f)
        #     stats = stats["pitch"] + stats["energy"][:2]
        stats = Define.ALLSTATS["global"][:6]

        # with open(os.path.join(figure_dir, f"{basename}.{name}.synth.npy"), 'wb') as f:
        #     np.save(f, mel_prediction.cpu().numpy())
        # fig = plot_mel(
        #     [
        #         (mel_prediction.cpu().numpy(), pitch, energy),
        #     ],
        #     stats,
        #     ["Synthetized Spectrogram"],
        # )
        # plt.savefig(os.path.join(figure_dir, f"{basename}.{name}.synth.png"))
        # plt.close()

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    wav_predictions = vocoder.infer(mel_predictions, max_wav_value, lengths=lengths)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, targets[0]):
        wavfile.write(os.path.join(audio_dir, f"{basename}.{name}.synth.wav"), sampling_rate, wav)


def loss2str(loss):
    return dict2str(loss2dict(loss))

def loss2dict(loss):
    tblog_dict = {
        "Total Loss"       : loss[0].item(),
        "Mel Loss"         : loss[1].item(),
        "Mel-Postnet Loss" : loss[2].item(),
        "Pitch Loss"       : loss[3].item(),
        "Energy Loss"      : loss[4].item(),
        "Duration Loss"    : loss[5].item(),
    }
    return tblog_dict

def dict2loss(tblog_dict):
    loss = (
        tblog_dict["Total Loss"],
        tblog_dict["Mel Loss"],
        tblog_dict["Mel-Postnet Loss"],
        tblog_dict["Pitch Loss"],
        tblog_dict["Energy Loss"],
        tblog_dict["Duration Loss"],
    )
    return loss

def dict2str(tblog_dict):
    message = ", ".join([f"{k}: {v:.4f}" for k, v in tblog_dict.items()])
    return message


def asr_loss2dict(loss):
    return {
        "Total Loss": loss[0].item(),
        "Phoneme Loss": loss[1].item(),
        "Cluster Loss": loss[2].item(),
    }


def pr_loss2dict(loss):
    return {
        "Total Loss": loss.item(),
    }


def dual_loss2dict(loss):
    tts_loss, asr_loss = loss
    tblog_dict = {
        "Total Loss"       : tts_loss[0].item() + asr_loss[0].item(),
        "Mel Loss"         : tts_loss[1].item(),
        "Mel-Postnet Loss" : tts_loss[2].item(),
        "Pitch Loss"       : tts_loss[3].item(),
        "Energy Loss"      : tts_loss[4].item(),
        "Duration Loss"    : tts_loss[5].item(),
        "Phoneme Loss"     : asr_loss[1].item(),
        "Cluster Loss"     : asr_loss[2].item(),
    }
    return tblog_dict
