<div align="center">

# VAT-SS: Investigation of Speech Separation Models Including Video Source

[\[🔥 VAT-SS Report\]](src/docs/paper.pdf)

</div>

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-extract-video-embeddings">How To Extract Video Embeddings</a> •
  <a href="#how-to-train">How To Train</a> •
  <a href="#how-to-evaluate">How To Evaluate</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

VAT-SS is Andrey-Vera-Teasgen-SpeechSeparation models family. This repository allows user to train and evaluate mentioned in report SS models.

> Pay attention that in all configs base model is state-of-the-art DPTN-AV-repack-by-teasgen, but you may use other SS models reported in the paper additionally (take a look at other configs)

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

   ```bash
   # create env
   conda create -n project_env python=3.10

   # activate env
   conda activate project_env
   ```

1. Install all required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Extract Video Embeddings
This section is mandatory for running Train and Evaluation script for Audio-Video models. Preliminary video embeddings extraction is necessary for speed up forward time.
```bash
bash download_lipreader.sh

python make_embeddings.py \
    --cfg_path src/lipreader/configs/lrw_resnet18_mstcn.json \
    --lipreader_path lrw_resnet18_mstcn_video.pth \
    --mouths_dir mouths \
    --embeds_dir embeddings
```
The embeddings will be saved to `--embeds_dir`. Please set correct path to your directory in all Hydra configs at Datasets level

## How To Train
You should have single A100-80gb GPU to exactly reproduce training, otherwise please implement and use gradient accumulation

To train a model, run the following commands and register in WandB:

Two-steps training:
```bash
python3 train.py -cn dptn_wav_av.yaml dataloader.batch_size=16 writer.run_name=av_dptn_wav_av_v1_video_tanh_gate
```

Moreover, training logs are available in WandBs

- DPRNN & DPTN https://wandb.ai/teasgen/ss/overview
- ConvTasNet https://wandb.ai/verabuylova-nes/ss/overview
- VoiceFilter & RTFS https://wandb.ai/aapetukhov-new-economic-school/ss?nw=nwuseraapetukhov

## How To Evaluate
Read How To Extract Video Embeddings section before

All generated texts will be saved into `data/saved/inferenced/<dataset part>` directory with corresponing names. Download SOTA pretrained model using
```bash
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1egOSgh3qaADxWpxd379nmhLrfZ-5xYEf' -O ./model.tar
tar xvf ./model.tar
```

To run inference and calculate metrics, provide custom dataset, change paths to WAVs and video embeddings in cmd arguments `datasets.val.audio_dir`, `datasets.val.embedding_dir` and run:
   ```bash
   python3 inference.py -cn inference_dptn_av.yaml dataloader.batch_size=32 inferencer.from_pretrained=model_best.pth datasets.val.part=null datasets.val.audio_dir=<PATH_TO_WAVS> datasets.val.embedding_dir=<PATH_TO_EMBEDDINGS>
   ```
   Set dataloader.batch_size not more than len(dataset)

   In case you don't have GT please change `device_tensors` in `inference_dptn_av.yaml` config to `device_tensors: ["mix_spectrogram", "mix", "s1_embedding", "s2_embedding"]`, following that metrics won't be calculated and only predictions will be saved.
   Or via cmd arguments: `inferencer.device_tensors="["mix_spectrogram","mix","s1_embedding","s2_embedding"]"`

   Use following command to run SiSNRi calculation on GT and predicted directories
   ```bash
   export PYTHONPATH=./
   python3 src/utils/eval_si_snri.py --predicts-dir <PATH_TO_PREDS> --gt-dir <PATH_TO_GTS>
   ```
   <PATH_TO_PREDS> is directory containing predicts file in .pth format
   <PATH_TO_GTS> is directory containing s1, s2, mix dirs


To evaluate the computational performance of the model, run:
   ```bash
   python3 profiler.py
   ```

Best model DPTN-AV-repack profiler results in Kaggle enviroment with P100 GPU:
| Metric                | Value               |
|-----------------------|---------------------|
| GFLOPs                | 108.556458241       |
| CUDA Memory           | 14378.582016        |
| Inference Time (Mean) | 0.09988968074321747 |
| Inference Time (Std)  | 0.04486224800348282 |
| Number of Parameters  | 40809590            |

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
