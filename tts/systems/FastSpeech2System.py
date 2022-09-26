import torch.nn as nn

import Define
from text.define import LANG_ID2SYMBOLS
from tts.models.FastSpeech2.embedding import MultilingualEmbedding
from tts.models.FastSpeech2.fastspeech2m import FastSpeech2
from tts.models.FastSpeech2.loss import FastSpeech2Loss
from .system import System
from tts.callbacks.FastSpeech2.saver import Saver


class FastSpeech2System(System):
    """
    Concrete class for multilingual FastSpeech2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.test_list = {
            "generate": self.generate_wavs,
        }

    def build_model(self):
        encoder_dim = self.model_config["transformer"]["encoder_hidden"]
        self.embedding_layer = MultilingualEmbedding(lang_id2symbols=LANG_ID2SYMBOLS, dim=encoder_dim)
        self.model = FastSpeech2(self.model_config, self.algorithm_config, binning_stats=Define.ALLSTATS["global"])
        self.loss_func = FastSpeech2Loss(self.model_config)

    def build_optimized_model(self):
        return nn.ModuleList([self.model, self.embedding_layer])
    
    def build_saver(self):
        saver = Saver(self.log_dir, self.result_dir, self.model_config)
        return saver

    def common_step(self, batch, batch_idx, train=True):
        """
        Data batch returned from collate:
            ids,
            raw_texts,
            speaker_args,
            torch.from_numpy(texts).long(),
            torch.from_numpy(text_lens),
            max(text_lens),
            torch.from_numpy(mels).float(),
            torch.from_numpy(mel_lens),
            max(mel_lens),
            torch.from_numpy(pitches).float(),
            torch.from_numpy(energies),
            torch.from_numpy(durations).long(),
            lang_ids,
        """
        emb_texts = self.embedding_layer(batch[3], lang_id=batch[12][0])
        output = self.model(batch[2], emb_texts, *(batch[4:-1]))
        loss = self.loss_func(batch[:-1], output)
        loss_dict = {
            "Total Loss"       : loss[0],
            "Mel Loss"         : loss[1],
            "Mel-Postnet Loss" : loss[2],
            "Pitch Loss"       : loss[3],
            "Energy Loss"      : loss[4],
            "Duration Loss"    : loss[5],
        }
            
        return loss_dict, output
    
    def synth_step(self, batch, batch_idx):
        """
        Synthesize without pitch and energy.
        """
        emb_texts = self.embedding_layer(batch[3], lang_id=batch[12][0])
        output = self.model(batch[2], emb_texts, *(batch[4:6]))
        return output

    def training_step(self, batch, batch_idx):
        train_loss_dict, output = self.common_step(batch, batch_idx, train=True)

        # Log metrics to CometLogger
        loss_dict = {f"Train/{k}": v.item() for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': train_loss_dict["Total Loss"], 'losses': train_loss_dict, 'output': output, '_batch': batch}

    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions = self.common_step(batch, batch_idx, train=False)
        synth_predictions = self.synth_step(batch, batch_idx)

        # Log metrics to CometLogger
        loss_dict = {f"Val/{k}": v.item() for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True)
        return {'loss': val_loss_dict["Total Loss"], 'losses': val_loss_dict, 'output': predictions, '_batch': batch, 'synth': synth_predictions}
    
    def test_step(self, batch, batch_idx):
        outputs = {}
        for test_name, test_fn in getattr(self, "test_list", {}).items(): 
            outputs[test_name] = test_fn(batch, batch_idx)

        return outputs

    def generate_wavs(self, batch, batch_idx):
        synth_predictions = self.synth_step(batch, batch_idx)
        return {'_batch': batch, 'synth': synth_predictions}
