import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import random
from tqdm import tqdm
from contextlib import contextmanager
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Agg")

import Define


@contextmanager
def seed_all(seed=None, devices=None):
    rstate = random.getstate()
    nstate = np.random.get_state()
    with torch.random.fork_rng(devices):
        random.seed(seed)
        np.random.seed(seed)
        if seed is None:
            seed = torch.seed()
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        yield
    random.setstate(rstate)
    np.random.set_state(nstate)


def get_mask_from_lengths(lengths, max_len=None, pad_to_multiple=1):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    if max_len % pad_to_multiple != 0:
        max_len += pad_to_multiple - max_len % pad_to_multiple
        assert max_len % pad_to_multiple == 0

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(Define.DEVICE)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max, energy_mean, energy_std = stats

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    fig.subplots_adjust(hspace=0.3)
    for i in range(len(data)):
        mel, pitch, energy = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


# Origin Author: Daniel Lin
def ssl_match_length(inputs, target_len: int):
    """
    Since the upstream extraction process can sometimes cause a mismatch
    between the seq lenth of inputs and labels:
    - if len(inputs) > len(labels), we truncate the final few timestamp of inputs to match the length of labels
    - if len(inputs) < len(labels), we duplicate the last timestep of inputs to match the length of labels
    Note that the length of labels should never be changed.
    Input is always SSL feature with shape (B, L, *dim).
    """
    factors = [1] * inputs.dim()
    input_len, label_len = inputs.size(1), target_len
    if input_len > label_len:
        inputs = inputs[:, :label_len, :]
    elif input_len < label_len:
        pad_vec = inputs[:, -1, :].unsqueeze(1)  # (batch_size, 1, *dim)
        factors[1] = label_len - input_len
        inputs = torch.cat((inputs, pad_vec.repeat(*factors)), dim=1)  # (batch_size, seq_len, *dim), where seq_len == target_len
    return inputs
