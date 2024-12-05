import argparse
import os
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio

from src.metrics.si_snri import SISNRiMetric

def load_audio(path, sr):
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
    target_sr = sr
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    return audio_tensor

def calculate(pred_dir: str, gt_dir: str, sr: int) -> dict[int, str]:
    metric = SISNRiMetric()

    path_s1 = Path(os.path.join(gt_dir, "s1"))
    si_snri = []
    for file_abs_path in tqdm(path_s1.iterdir()):
        original_name_wav = file_abs_path.stem + ".wav"
        original_name_pth = file_abs_path.stem + ".pth"
        s1_gt = file_abs_path
        mix = os.path.join(gt_dir, "mix", original_name_wav)
        s2_gt = os.path.join(gt_dir, "s2", original_name_wav)
        pred = torch.load(os.path.join(pred_dir, original_name_pth))
        si_snri.append(metric(
            mix=load_audio(mix, sr),
            s1=load_audio(s1_gt, sr),
            s2=load_audio(s2_gt, sr),
            s1_pred=pred["s1_pred"][None, :].cpu(),
            s2_pred=pred["s2_pred"][None, :].cpu(),
        ).item())
    return {
        "SiSNRi": sum(si_snri) / len(si_snri),
    }
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicts-dir', type=str, help='directory with prediction s1 / s2 dirs')
    parser.add_argument('--gt-dir', type=str, help='directory with predicted .pth files')
    parser.add_argument('--sr', default=16000, type=int, help='target sr')
    args = parser.parse_args()

    results = calculate(args.predicts_dir, args.gt_dir, args.sr)
    for x, y in results.items():
        print(x, y, sep=": ")