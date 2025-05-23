import json
import os
from pathlib import Path

import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class SSDataset(BaseDataset):
    def __init__(self, part="train", audio_dir=None, video_dir=None, embedding_dir=None, *args, **kwargs):
        """
        Args:
            part (str): partition name
        """
        if audio_dir is None:
            self._audio_dir = ROOT_PATH / "audio"
        else:
            self._audio_dir = Path(audio_dir)

        if video_dir is None:
            self._video_dir = ROOT_PATH / "mouth"
        else:
            self._video_dir = Path(video_dir)

        if embedding_dir is None:
            self._embedding_dir = ROOT_PATH / "embedding"
        else:
            self._embedding_dir = Path(embedding_dir)

        if self._video_dir.exists():
            self.contains_video = True
        else:
            self.contains_video = False

        if self._embedding_dir.exists():
            self.contains_embedding = True
        else:
            self.contains_embedding = False

        if part is None:
            part = "custom"
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._audio_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        if part == "custom":
            split_dir = self._audio_dir
        else:
            split_dir = self._audio_dir / part
        mix_split_dir = split_dir / "mix"

        for wavname in os.listdir(mix_split_dir):
            id1, id2 = wavname.replace(".wav", "").split("_")

            mix_wav_path = mix_split_dir / wavname
            s1_wav_path = None
            s2_wav_path = None
            s1_video_path = None
            s2_video_path = None
            s1_embedding_path = None
            s2_embedding_path = None

            if os.path.exists(split_dir / "s1"):
                s1_wav_path = split_dir / "s1" / wavname
                s2_wav_path = split_dir / "s2" / wavname

                s1_wav_path = str(s1_wav_path.absolute().resolve())
                s2_wav_path = str(s2_wav_path.absolute().resolve())

            if self.contains_video:
                s1_video_path = self._video_dir / f"{id1}.npz"
                s2_video_path = self._video_dir / f"{id2}.npz"

                s1_video_path = str(s1_video_path.absolute().resolve())
                s2_video_path = str(s2_video_path.absolute().resolve())

            if self.contains_embedding:
                s1_embedding_path = self._embedding_dir / f"{id1}.npz"
                s2_embedding_path = self._embedding_dir / f"{id2}.npz"

                s1_embedding_path = str(s1_embedding_path.absolute().resolve())
                s2_embedding_path = str(s2_embedding_path.absolute().resolve())

            t_info = torchaudio.info(str(mix_wav_path))
            length = t_info.num_frames / t_info.sample_rate

            index.append(
                {
                    "mix_wav_path": str(mix_wav_path.absolute().resolve()),
                    "s1_wav_path": s1_wav_path,
                    "s2_wav_path": s2_wav_path,
                    "s1_video_path": s1_video_path,
                    "s2_video_path": s2_video_path,
                    "s1_embedding_path": s1_embedding_path,
                    "s2_embedding_path": s2_embedding_path,
                    "audio_len": length,
                }
            )

        return index
