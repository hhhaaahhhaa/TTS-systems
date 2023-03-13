import torch


def get_mask_from_lengths(lengths, max_len=None, pad_to_multiple=1):
    """
    Unmasked: 0, Masked: 1
    """
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    if max_len % pad_to_multiple != 0:
        max_len += pad_to_multiple - max_len % pad_to_multiple
        assert max_len % pad_to_multiple == 0

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def segment2duration(segment, fp):
    res = []
    for (s, e) in segment:
        res.append(
            int(
                round(round(e * 1 / fp, 4))  # avoid floating point numercial issue
                - round(round(s * 1 / fp, 4))
            )
        )
    return res


def expand(seq, dur):
    assert len(seq) == len(dur)
    res = []
    for (x, d) in zip(seq, dur):
        if d > 0:
            res.extend([x] * d)
    return res
