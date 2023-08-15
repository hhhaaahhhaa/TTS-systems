# TTS Systems - Pytorch Lightning Implementation of common architectures

Pytorch Lightning implementation for various commonly used TTS systems.

- Tacotron2
- FastSpeech2
- TBD

This project is based on
- [BogiHsu/Tacotron2-PyTorch](https://github.com/BogiHsu/Tacotron2-PyTorch
)
- [hhhaaahhhaa/dlhlp-lib](https://github.com/hhhaaahhhaa/dlhlp-lib)
- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)

Feel free to use/modify the code.

Require python version 3.8. You can install the Python dependencies with
```
pip install -r requirements.txt
```
# Preprocessing

Support various TTS datasets(LJSpeech/LibriTTS/AISHELL-3/...). ``DATASET_TAG`` can be ``LJSpeech``, ``LibriTTS``, ``AISHELL-3``. More tags can be found in ``Parsers/__init__.py``.


Alignments(TextGrid) of the supported datasets are provided [here](https://drive.google.com/drive/folders/1OyEh823slo4Taw9A-zlC9ruS45hz8Y81?usp=sharing). Unzip the files to ``preprocessed_data/[DATASET_NAME]/TextGrid/``.

To preprocess the dataset, execute the following scripts in order. ``[raw_dir]`` is the local path you place the downloaded speech dataset(LJSpeech/LibriTTS...).

```python=
python preprocess.py [raw_dir] [preprocessed_dir] --dataset LJSpeech --parse_raw  # unify formats from different TTS datasets
python preprocess.py [raw_dir] [preprocessed_dir] --dataset LJSpeech --preprocess  # main script
python clean.py [preprocessed_dir] [clean_results_path]  # filtering
python preprocess.py [raw_dir] [preprocessed_dir] --dataset [DATASET_TAG] --create_dataset [clean_results_path]  # split dataset
```

Take LJSpeech as an example:
```python=
python preprocess.py [raw_dir] preprocessed_data/LJSpeech-1.1 --dataset LJSpeech --parse_raw
python preprocess.py [raw_dir] preprocessed_data/LJSpeech-1.1 --dataset LJSpeech --preprocess
python clean.py preprocessed_data/LJSpeech-1.1 data_config/LJSpeech-1.1/clean.json
python preprocess.py [raw_dir] preprocessed_data/LJSpeech-1.1 --dataset LJSpeech --create_dataset data_config/LJSpeech-1.1/clean.json
```

To preprocess different datasets or apply custom preprocessing/dataset filtering/dataset splitting, feel free to modify the script.

# Tacotron2

Support LJSpeech, LibriTTS. Take LJSpeech as example.

## Training

Train your model with

```
python tacotron2_train.py -s train -d data_config/LJSpeech-1.1 -m config/Tacotron2/model/base.yaml -t config/Tacotron2/train/baseline.yaml -a config/Tacotron2/algorithm/baseline.yaml -n [exp_name]
```

For resuming training from a checkpoint, run the following command.

```
python tacotron2_train.py -s train -d data_config/LJSpeech-1.1 -m config/Tacotron2/model/base.yaml -t config/Tacotron2/train/baseline.yaml -a config/Tacotron2/algorithm/baseline.yaml -n [exp_name] -pre [ckpt_path]
```

Remember to change to your local path inside data_config. The results are under ``output/[exp_name]``. For multispeaker dataset, use ``base-multispeaker.yaml`` as model_config.

## Inference

For synthesizing wav files, modify parameters inside ``tacotron2_inference.py`` and run the script.
```
python tacotron2_inference.py
```

# FastSpeech2

Support LJSpeech, LibriTTS, AISHELL-3. Take LJSpeech as example.

## Training

Train your model with

```
python fastspeech2_train.py -s train -d data_config/LJSpeech-1.1 -m config/FastSpeech2/model/base.yaml -t config/FastSpeech2/train/baseline.yaml -a config/FastSpeech2/algorithm/baseline.yaml -n [exp_name]
```

For resuming training from a checkpoint, run the following command.

```
python fastspeech2_train.py -s train -d data_config/LJSpeech-1.1 -m config/FastSpeech2/model/base.yaml -t config/FastSpeech2/train/baseline.yaml -a config/FastSpeech2/algorithm/baseline.yaml -n [exp_name] -pre [ckpt_path]
```

Remember to change to your local path inside data_config. The results are under ``output/[exp_name]``. For multispeaker dataset, use ``base-multispeaker.yaml`` as model_config.

## Inference

For synthesizing wav files, modify parameters inside ``fastspeech2_inference.py`` and run the script.
```
python fastspeech2_inference.py
```
You can control pitch, energy, duration predictions of FastSpeech2.

# Vocoder

``dlhlp_lib`` implements Griffin-lim, MelGAN(deprecated), HifiGAN vocoder. Vocoder classes share the same inference interface. Input is a batch of Mel-Spectrograms with shape ``(batch_size, n_channels, time_step)``.

```python=
vocoder_cls = get_vocoder("HifhGAN")
vocoder = vocoder_cls()
wav = vocoder.infer(mel.unsqueeze(0), lengths=None)[0]  # inference single spectrogram
```


# TensorBoard

Use
```
tensorboard --logdir output/[exp_name]/log/tb
```
to serve TensorBoard on your localhost.
The loss curves, synthesized Mel-Spectrograms, and audios are shown.

