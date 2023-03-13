import torch
from torch import nn
from math import sqrt

from dlhlp_lib.audio import AUDIO_CONFIG
import Define
from .hparams import model_config2hparams
from .hparams import hparams as hps
from .model import Encoder, Decoder, Postnet
from tts.utils.tool import get_mask_from_lengths


class Tacotron2(nn.Module):
	def __init__(self, model_config):
		super(Tacotron2, self).__init__()
		model_config2hparams(model_config)
		self.num_mels = AUDIO_CONFIG["mel"]["n_mel_channels"]
		self.mask_padding = hps.mask_padding
		self.n_frames_per_step = hps.n_frames_per_step
		self.embedding = None  # required to be set
		self.encoder = Encoder()
		self.decoder = Decoder()
		self.postnet = Postnet()

		self.sid_emb = None
		if model_config["multi_speaker"]:
			self.sid_emb = nn.Embedding(
				model_config["n_speaker"],
				hps.encoder_embedding_dim,
			)

	def parse_batch(self, batch):
		text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spks, lang_ids = batch
		max_len = torch.max(input_lengths.data).item()

		return (
			(text_padded, input_lengths, mel_padded, max_len, output_lengths, spks, lang_ids),
			(mel_padded, gate_padded, output_lengths))

	def parse_output(self, outputs, output_lengths=None):
		if output_lengths is not None:
			mask = get_mask_from_lengths(output_lengths, pad_to_multiple=hps.n_frames_per_step) # (B, T)
			mask = mask.expand(hps.num_mels, mask.size(0), mask.size(1)) # (80, B, T)
			mask = mask.permute(1, 0, 2) # (B, 80, T)
			
			outputs[0].data.masked_fill_(mask, 0.0) # (B, 80, T)
			outputs[1].data.masked_fill_(mask, 0.0) # (B, 80, T)
			slice = torch.arange(0, mask.size(2), hps.n_frames_per_step)
			outputs[2].data.masked_fill_(mask[:, 0, slice], 1e3)  # gate energies (B, T//n_frames_per_step)
		return outputs

	def forward(self, inputs):
		text_inputs, text_lengths, mels, max_len, output_lengths, spks, lang_ids = inputs
		text_lengths, output_lengths = text_lengths.data, output_lengths.data

		embedded_inputs = self.embedding(text_inputs).transpose(1, 2)  # currently support monolingual training only

		encoder_outputs = self.encoder(embedded_inputs, text_lengths)

		if self.sid_emb is not None:
			sid_embs = self.sid_emb(spks)
			encoder_outputs = encoder_outputs + sid_embs.unsqueeze(1)

		mel_outputs, gate_outputs, alignments = self.decoder(
			encoder_outputs, mels, memory_lengths=text_lengths)

		mel_outputs_postnet = self.postnet(mel_outputs)
		mel_outputs_postnet = mel_outputs + mel_outputs_postnet

		return self.parse_output(
			[mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
			output_lengths)

	def inference(self, inputs, spks, lang_ids):
		embedded_inputs = self.embedding(inputs, lang_ids[0]).transpose(1, 2)
		encoder_outputs = self.encoder.inference(embedded_inputs)

		if self.sid_emb is not None:
			sid_embs = self.sid_emb(spks)
			encoder_outputs = encoder_outputs + sid_embs.unsqueeze(1)

		mel_outputs, gate_outputs, alignments = self.decoder.inference(
			encoder_outputs)

		mel_outputs_postnet = self.postnet(mel_outputs)
		mel_outputs_postnet = mel_outputs + mel_outputs_postnet

		outputs = self.parse_output(
			[mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

		return outputs

	def teacher_infer(self, inputs, mels, lang_ids):
		il, _ =  torch.sort(torch.LongTensor([len(x) for x in inputs]),
							dim = 0, descending = True)
		text_lengths = il.to(Define.DEVICE)

		embedded_inputs = self.embedding(inputs, lang_ids[0]).transpose(1, 2)

		encoder_outputs = self.encoder(embedded_inputs, text_lengths)

		mel_outputs, gate_outputs, alignments = self.decoder(
			encoder_outputs, mels, memory_lengths=text_lengths)
		
		mel_outputs_postnet = self.postnet(mel_outputs)
		mel_outputs_postnet = mel_outputs + mel_outputs_postnet

		return self.parse_output(
			[mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

	# def infer(self, text, spk=None):
	# 	sequence = text_to_sequence(text, hps.text_cleaners)
	# 	sequence = torch.IntTensor(sequence)[None, :].long().to(Define.DEVICE)

	# 	if spk is None:
	# 		assert self.sid_emb is None, "Model does not support multispeaker!"
		

	# 	mel_outputs, mel_outputs_postnet, _, alignments = self.inference(sequence, spk)

	# 	return (mel_outputs, mel_outputs_postnet, alignments)
