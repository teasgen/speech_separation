import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from src.utils.init_utils import init_lipreader
from src.lipreader.lipreading.dataloaders import get_preprocessing_pipelines


def main(
    cfg_path: str,
    lipreader_path: str,
    mouths_dir: str,
    embeds_dir: str,
):
    """runs in terminal with the following command:

    #!/bin/bash
    python make_embeddings.py \
        --cfg_path src/lipreader/configs/lrw_resnet18_mstcn.json \
        --lipreader_path lrw_resnet18_mstcn_video.pth \
        --mouths_dir mouths \
        --embeds_dir embeddings

    WARNING: select your own mouths_dir and embeds_dir before running script.
    You don't want mouths dir to be empty
    """
    
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config path not found: {cfg_path}")
    if not os.path.exists(lipreader_path):
        raise FileNotFoundError(f"Lipreader path not found: {lipreader_path}")
    if not os.path.isdir(mouths_dir):
        raise NotADirectoryError(f"Input directory not found: {mouths_dir}")
    if not os.path.exists(embeds_dir):
        os.makedirs(embeds_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("MOST LIPREADERS RUN ON CUDA, NOT ON CPU")
    print("DEVICE SELECTED: ", device)

    lipreader = init_lipreader(cfg_path, lipreader_path).to(device)
    lipreader.eval()

    # preprocessing
    preprocessing_func = get_preprocessing_pipelines(modality="video")["test"]

    os.makedirs(embeds_dir, exist_ok=True)

    counter = 0
    for filename in tqdm(os.listdir(mouths_dir), desc="mouth files"):
        file_path = os.path.join(mouths_dir, filename)

        if os.path.isfile(file_path):
            data = np.load(file_path)
            video = torch.FloatTensor(data["data"]).to(device)  # [T, H, W]
            s_data = preprocessing_func(video)  # [T, H, W]

            # lipreader
            s_data = s_data.unsqueeze(0).unsqueeze(1)
            embed = lipreader(s_data, lengths=[50]).squeeze(0).transpose(0, 1)  # [1, 512, 50]
            embedding_np = embed.detach().cpu().numpy()

            embed_file_path = os.path.join(embeds_dir, filename)
            np.savez_compressed(embed_file_path, embedding=embedding_np)
            counter += 1

    print("SUCCESS: embeddings saved to ", embeds_dir)
    print("TOTAL: ", counter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate lipreader embeddings")
    parser.add_argument("--cfg_path", type=str, default="src/lipreader/configs/lrw_resnet18_mstcn.json", help="Config path")
    parser.add_argument("--lipreader_path", type=str, default="lrw_resnet18_mstcn_video.pth", help="Lipreader .pth file path")
    parser.add_argument("--mouths_dir", type=str, default="mouths", help="Dir with mouths")
    parser.add_argument("--embeds_dir", type=str, default="embeddings", help="Dir where to put embeddings")

    args = parser.parse_args()

    main(args.cfg_path, args.lipreader_path, args.mouths_dir, args.embeds_dir)
