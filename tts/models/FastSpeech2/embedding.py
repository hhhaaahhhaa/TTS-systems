import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilingualEmbedding(nn.Module):
    def __init__(self, lang_id2symbols, dim: int, padding_idx: int=0):
        super().__init__()
        self.lang_id2symbols = lang_id2symbols
        self.dim = dim
        self.padding_idx = padding_idx

        self.tables = nn.ParameterDict()
        for lang_id, v in lang_id2symbols.items():
            if len(v) > 0:
                w_init = torch.randn(len(v), dim)
                w_init[padding_idx].fill_(0)
                self.tables[f"table-{lang_id}"] = nn.Parameter(w_init)

    def forward(self, x, lang_id: str=""):
        if lang_id == "":
            concat_tables = torch.cat([p for p in self.tables.values()], dim=0)
            return F.embedding(x, concat_tables, padding_idx=self.padding_idx)
        return F.embedding(x, self.tables[f"table-{lang_id}"], padding_idx=self.padding_idx)
