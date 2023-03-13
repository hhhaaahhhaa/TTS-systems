import torch.nn as nn

from tts.models.embedding import MultilingualEmbedding
from tts.models.Tacotron2.tacotron import Tacotron2
from tts.models.Tacotron2.model import Tacotron2Loss
from .system import System
from tts.callbacks.Tacotron2.saver import Saver
from tts.build import build_id2symbols


class Tacotron2System(System):
    """
    Concrete class for multilingual Tacotron2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bs = self.train_config["optimizer"]["batch_size"]

    def build_model(self):
        dim = self.model_config["tacotron2"]["symbols_embedding_dim"]
        embedding_layer = MultilingualEmbedding(id2symbols=build_id2symbols(self.data_configs), dim=dim)
        self.model = Tacotron2(self.model_config)
        self.model.embedding = embedding_layer
        self.loss_func = Tacotron2Loss()

    def build_optimized_model(self):
        return nn.ModuleList([self.model])
    
    def build_saver(self):
        self.saver = Saver(self.data_configs, self.model_config, self.log_dir, self.result_dir)
        return self.saver

    def common_step(self, batch, batch_idx, train=True):
        x, y = self.model.parse_batch(batch)
        output = self.model(x)
        loss, items = self.loss_func(output, y)
        
        loss_dict = {
            "Total Loss": loss,
            "Mel Loss": items[0],
            "Gate Loss": items[1],
        }
            
        return loss_dict, output

    def training_step(self, batch, batch_idx):
        train_loss_dict, output = self.common_step(batch, batch_idx, train=True)
        loss = train_loss_dict["Total Loss"]
        train_loss_dict["Total Loss"] = train_loss_dict["Total Loss"].item()

        # Log metrics to Logger
        loss_dict = {f"Train/{k}": v for k, v in train_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': loss, 'losses': train_loss_dict, 'output': output, '_batch': batch}

    def validation_step(self, batch, batch_idx):
        val_loss_dict, predictions = self.common_step(batch, batch_idx, train=False)
        loss = val_loss_dict["Total Loss"]
        val_loss_dict["Total Loss"] = val_loss_dict["Total Loss"].item()

        # Log metrics to Logger
        loss_dict = {f"Val/{k}": v for k, v in val_loss_dict.items()}
        self.log_dict(loss_dict, sync_dist=True, batch_size=self.bs)
        return {'loss': loss, 'losses': val_loss_dict, 'output': predictions, '_batch': batch}
    
    def test_step(self, batch, batch_idx):
        outputs = {}
        for test_name, test_fn in getattr(self, "test_list", {}).items(): 
            outputs[test_name] = test_fn(batch, batch_idx)

        return outputs
