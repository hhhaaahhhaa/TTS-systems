import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import merge_dicts
from pytorch_lightning.utilities import rank_zero_only

from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.vocoders import get_vocoder
from dlhlp_lib.utils import to_arr

import Define
from tts.callbacks.BaseSaver import BaseSaver
from .plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy


CSV_COLUMNS = None
COL_SPACE = None


def set_format(keys: List[str]):
    global CSV_COLUMNS, COL_SPACE
    CSV_COLUMNS = keys
    COL_SPACE = [len(col) for col in ["200000", "Validation"]+CSV_COLUMNS]


class Saver(BaseSaver):

    def __init__(self, data_configs, model_config, log_dir, result_dir):
        super().__init__(log_dir, result_dir)
        self.data_configs = data_configs
        self.model_config = model_config
        self.sr = AUDIO_CONFIG["audio"]["sampling_rate"]
        
        self.val_loss_dicts = []
        self.log_loss_dicts = []
        vocoder_cls = get_vocoder(self.model_config["vocoder"]["model"])
        self.vocoder = vocoder_cls().cuda()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss_dict = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']

        step = pl_module.global_step + 1

        if Define.LOGGER != "":
            logger = pl_module.logger

        # Synthesis one sample and log to Logger
        if Define.LOGGER != "":
            if step % pl_module.train_config["step"]["synth_step"] == 0 and pl_module.local_rank == 0:
                mel_outputs = to_arr(output[0][0])
                mel_outputs_postnet = to_arr(output[1][0])
                alignments = to_arr(output[3][0]).T

                self.log_image(logger, "Train", step, "", "align", plot_alignment_to_numpy(alignments))
                self.log_image(logger, "Train", step, "", "mel", plot_spectrogram_to_numpy(mel_outputs))
                self.log_image(logger, "Train", step, "", "mel_post", plot_spectrogram_to_numpy(mel_outputs_postnet))

                wav_reconstruction = self.vocoder.infer(output[0][0].unsqueeze(0))[0]
                wav_prediction = self.vocoder.infer(output[1][0].unsqueeze(0))[0]
                self.log_audio(logger, "Training", step, "", "pred", wav_reconstruction, self.sr)
                self.log_audio(logger, "Training", step, "", "pred_postnet", wav_prediction, self.sr)

        # Log message to log.txt and print to stdout
        if step % trainer.log_every_n_steps == 0 and pl_module.local_rank == 0:
            if CSV_COLUMNS is None:
                set_format(list(loss_dict.keys()))
            loss_dict.update({"Step": step, "Stage": "Training"})
            df = pd.DataFrame([loss_dict], columns=["Step", "Stage"]+CSV_COLUMNS)
            if len(self.log_loss_dicts)==0:
                tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE))
            else:
                tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE).split('\n')[-1])
            self.log_loss_dicts.append(loss_dict)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_loss_dicts = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss_dict = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']
        
        step = pl_module.global_step + 1
        if Define.LOGGER != "":
            logger = pl_module.logger

        if CSV_COLUMNS is None:
            set_format(list(loss_dict.keys()))
        self.val_loss_dicts.append(loss_dict)

        # Log loss for each sample to csv files
        self.save_csv("Validation", step, 0, loss_dict)

        figure_dir = os.path.join(self.result_dir, "figure")
        audio_dir = os.path.join(self.result_dir, "audio")
        os.makedirs(figure_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

        # Log figure/audio to logger + save audio
        if Define.LOGGER != "":
            if batch_idx == 0 and pl_module.local_rank == 0:
                mel_outputs = to_arr(output[0][0])
                mel_outputs_postnet = to_arr(output[1][0])
                alignments = to_arr(output[3][0]).T

                self.log_image(logger, "Validation", step, "", "align", plot_alignment_to_numpy(alignments))
                self.log_image(logger, "Validation", step, "", "mel", plot_spectrogram_to_numpy(mel_outputs))
                self.log_image(logger, "Validation", step, "", "mel_post", plot_spectrogram_to_numpy(mel_outputs_postnet))

                wav_reconstruction = self.vocoder.infer(output[0][0].unsqueeze(0))[0]
                wav_prediction = self.vocoder.infer(output[1][0].unsqueeze(0))[0]
                self.log_audio(logger, "Validation", step, "", "pred", wav_reconstruction, self.sr)
                self.log_audio(logger, "Validation", step, "", "pred_postnet", wav_prediction, self.sr)

    def on_validation_epoch_end(self, trainer, pl_module):
        loss_dict = merge_dicts(self.val_loss_dicts)
        step = pl_module.global_step + 1

        # Log total loss to log.txt and print to stdout
        loss_dict.update({"Step": step, "Stage": "Validation"})
        # To stdout
        df = pd.DataFrame([loss_dict], columns=["Step", "Stage"]+CSV_COLUMNS)
        if len(self.log_loss_dicts)==0:
            tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE))
        else:
            tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE).split('\n')[-1])
        # To file
        self.log_loss_dicts.append(loss_dict)
        log_file_path = os.path.join(self.log_dir, 'log.txt')
        df = pd.DataFrame(self.log_loss_dicts, columns=["Step", "Stage"]+CSV_COLUMNS).set_index("Step")
        df.to_csv(log_file_path, mode='a', header=not os.path.exists(log_file_path), index=True)
        
        # Reset
        self.log_loss_dicts = []
