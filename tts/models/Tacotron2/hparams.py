from dlhlp_lib.audio import AUDIO_CONFIG


def model_config2hparams(model_config):
    """
    Convert unified format into BogiHsu's Tacotron2 format
    """
    for k, v in model_config["tacotron2"].items():
        setattr(hparams, k, v)


class hparams:
    num_mels = AUDIO_CONFIG["mel"]["n_mel_channels"]
    symbols_embedding_dim = 512
    mask_padding = True

    # Encoder parameters
    encoder_kernel_size = 5
    encoder_n_convolutions = 3
    encoder_embedding_dim = 512

    # Decoder parameters
    n_frames_per_step = 3
    decoder_rnn_dim = 1024
    prenet_dim = 256
    max_decoder_ratio = 10
    gate_threshold = 0.5
    p_attention_dropout = 0.1
    p_decoder_dropout = 0.1

    # Attention parameters
    attention_rnn_dim = 1024
    attention_dim = 128

    # Location Layer parameters
    attention_location_n_filters = 32
    attention_location_kernel_size = 31

    # Mel-post processing network parameters
    postnet_embedding_dim = 512
    postnet_kernel_size = 5
    postnet_n_convolutions = 5
