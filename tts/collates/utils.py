import numpy as np
import torch

from tts.utils.tool import pad_1D, pad_2D


def reprocess(data, idxs, mode="sup"):
    """
    Pad data and calculate length of data.
    Inference version has only text and speaker data.
    Args:
        mode: "sup" or "inference".
    """
    ids = [data[idx]["id"] for idx in idxs]
    lang_ids = [data[idx]["lang_id"] for idx in idxs]
    speakers = [data[idx]["speaker"] for idx in idxs]
    speakers = np.array(speakers)

    if mode in ["sup", "inference"]:
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        text_lens = np.array([text.shape[0] for text in texts])
        texts = pad_1D(texts)

    if mode in ["sup"]:
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        mel_lens = np.array([mel.shape[0] for mel in mels])

    if mode in ["sup"]:
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

    speaker_args = torch.from_numpy(speakers).long()

    if mode == "sup":
        return (
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
            lang_ids
        )
    elif mode == "inference":
        return (
            ids,
            raw_texts,
            speaker_args,
            torch.from_numpy(texts).long(),
            torch.from_numpy(text_lens),
            max(text_lens),
            None, 
            None, 
            None,
            None,
            None,
            None,
            lang_ids,
        )
    else:
        raise NotImplementedError
