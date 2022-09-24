from .stft import TacotronSTFT
from .tools import get_mel_from_wav, inv_mel_spec


AUDIO_CONFIG = {
    "audio": {
        "sampling_rate": 22050,
    },
    "stft": {
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024
    },
    "mel": {
        "n_mel_channels": 80,
        "mel_fmin": 0,
        "mel_fmax": None,
    },
}

STFT = TacotronSTFT(
    AUDIO_CONFIG["stft"]["filter_length"],
    AUDIO_CONFIG["stft"]["hop_length"],
    AUDIO_CONFIG["stft"]["win_length"],
    AUDIO_CONFIG["mel"]["n_mel_channels"],
    AUDIO_CONFIG["audio"]["sampling_rate"],
    AUDIO_CONFIG["mel"]["mel_fmin"],
    AUDIO_CONFIG["mel"]["mel_fmax"],
)


def overwrite_audio_config(config):
    global AUDIO_CONFIG
    global STFT

    AUDIO_CONFIG.update(config)
    STFT = TacotronSTFT(
        AUDIO_CONFIG["stft"]["filter_length"],
        AUDIO_CONFIG["stft"]["hop_length"],
        AUDIO_CONFIG["stft"]["win_length"],
        AUDIO_CONFIG["mel"]["n_mel_channels"],
        AUDIO_CONFIG["audio"]["sampling_rate"],
        AUDIO_CONFIG["mel"]["mel_fmin"],
        AUDIO_CONFIG["mel"]["mel_fmax"],
    )
