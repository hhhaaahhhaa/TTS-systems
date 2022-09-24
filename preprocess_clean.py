import torch
import s3prl.hub as hub
from tqdm import tqdm

from Parsers.parser import DataParser
from lightning.utils.tool import numpy_exist_nan, torch_exist_nan


def check_existence_and_nan(queries, data_parser: DataParser):
    res = []
    for query in tqdm(queries):
        try:
            assert not numpy_exist_nan(data_parser.mfa_duration.read_from_query(query))
            assert not numpy_exist_nan(data_parser.unsup_duration.read_from_query(query))
            assert not numpy_exist_nan(data_parser.mel.read_from_query(query))
            assert not numpy_exist_nan(data_parser.interpolate_pitch.read_from_query(query))
            assert not numpy_exist_nan(data_parser.energy.read_from_query(query))
            assert not numpy_exist_nan(data_parser.spk_ref_mel_slices.read_from_query(query))
            res.append(query)
        except:
            print("NaN in feature or feature does not exist:")
            print(query)
    return res


def check_nan_ssl_feature(queries, data_parser: DataParser, extractor, batch_size=12):
    res = []
    wavs = []
    q_temp = []
    for i, query in tqdm(enumerate(queries)):
        try:
            wav = data_parser.wav_trim_16000.read_from_query(query)
        except:
            continue
        wavs.append(torch.from_numpy(wav).float().cuda())
        q_temp.append(query)
        if (i + 1) % batch_size == 0 or i + 1 == len(queries):
            with torch.no_grad():
                representation = extractor(wavs)
                representation = torch.stack(representation["hidden_states"], dim=1)  # bs, layer, L, dim
                for r, q in zip(representation, q_temp):
                    try:
                        assert not torch_exist_nan(r.detach().cpu())
                    except:
                        print("NaN in SSL feature:")
                        print(q)
                    res.append(q)
            wavs = []
            q_temp = []
    return res


def clean_txt(filename, output_path, data_parser: DataParser):
    # Read cache for efficiency
    data_parser.mfa_duration.read_all(refresh=True)
    data_parser.unsup_duration.read_all(refresh=True)
    data_parser.interpolate_pitch.read_all(refresh=True)
    data_parser.energy.read_all(refresh=True)

    queries = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            n, s, t, r = line.strip("\n").split("|")    
            queries.append({
                "spk": s,
                "basename": n,
                "_line": line,
            })
    # For debug usage
    # queries = queries[:100] 

    print("Remaining: ", len(queries))
    queries = check_existence_and_nan(queries, data_parser)
    print("Remaining: ", len(queries))

    model = getattr(hub, 'hubert_large_ll60k')().cuda()
    queries = check_nan_ssl_feature(queries,  data_parser, model)
    print("Remaining: ", len(queries))

    model = getattr(hub, 'wav2vec2_large_ll60k')().cuda()
    queries = check_nan_ssl_feature(queries,  data_parser, model)
    print("Remaining: ", len(queries))

    model = getattr(hub, 'wav2vec2_xlsr')().cuda()
    queries = check_nan_ssl_feature(queries,  data_parser, model)
    print("Remaining: ", len(queries))

    with open(output_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(q["_line"]) 


if __name__ == "__main__":
    # parser = DataParser("./preprocessed_data/LibriTTS")
    # clean_txt("./_data/LibriTTS/train-clean-100-clean.txt", "./_data/LibriTTS/train.txt", parser)
    # clean_txt("./_data/LibriTTS/dev-clean-clean.txt", "./_data/LibriTTS/val.txt", parser)
    # clean_txt("./_data/LibriTTS/test-clean-clean.txt", "./_data/LibriTTS/test.txt", parser)

    # parser = DataParser("./preprocessed_data/AISHELL-3")
    # clean_txt("./_data/AISHELL-3/train-clean.txt", "./_data/AISHELL-3/train.txt", parser)
    # clean_txt("./_data/AISHELL-3/val-clean.txt", "./_data/AISHELL-3/val.txt", parser)

    # parser = DataParser("./preprocessed_data/kss")
    # clean_txt("./_data/kss/train-clean.txt", "./_data/kss/train.txt", parser)
    # clean_txt("./_data/kss/val-clean.txt", "./_data/kss/val.txt", parser)

    # parser = DataParser("./preprocessed_data/CSS10/german")
    # clean_txt("./_data/CSS10/german/train-clean.txt", "./_data/CSS10/german/train.txt", parser)
    # clean_txt("./_data/CSS10/german/val-clean.txt", "./_data/CSS10/german/val.txt", parser)

    # parser = DataParser("./preprocessed_data/JSUT")
    # clean_txt("./_data/JSUT/train-clean.txt", "./_data/JSUT/train.txt", parser)
    # clean_txt("./_data/JSUT/val-clean.txt", "./_data/JSUT/val.txt", parser)

    parser = DataParser("./preprocessed_data/GlobalPhone/french")
    clean_txt("./_data/GlobalPhone/french/train-clean.txt", "./_data/GlobalPhone/french/train.txt", parser)
    clean_txt("./_data/GlobalPhone/french/val-clean.txt", "./_data/GlobalPhone/french/val.txt", parser)
