raw_dir=""
python preprocess.py $raw_dir preprocessed_data/GenAI --dataset GenAI --parse_raw --preprocess
python clean.py preprocessed_data/GenAI data_config/GenAI/clean.json
python preprocess.py $raw_dir preprocessed_data/GenAI --dataset GenAI --create_dataset data_config/GenAI/clean.json


# python fastspeech2_train.py -s train -d data_config/GenAI -m config/FastSpeech2/model/base.yaml -t config/FastSpeech2/train/baseline.yaml -a config/FastSpeech2/algorithm/unit.yaml -n u2s-debug