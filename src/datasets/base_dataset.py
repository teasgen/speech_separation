import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from src.encoders.stft import BaseEncoder

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index,
        limit=None,
        target_sr=16000,
        encoder: BaseEncoder=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            target_sr (int): supported sample rate.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)
        self.target_sr = target_sr

        if encoder is not None:
            self.encoder = encoder

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index

        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        mix_wav_path = data_dict["mix_wav_path"]
        mix_audio = self.load_audio(mix_wav_path)
        s1_audio = None
        s2_audio = None
        s1_video = None
        s2_video = None
        s1_embedding = None
        s2_embedding = None

        if data_dict["s1_wav_path"] is not None:
            s1_wav_path = data_dict["s1_wav_path"]
            s1_audio = self.load_audio(s1_wav_path)

            s2_wav_path = data_dict["s2_wav_path"]
            s2_audio = self.load_audio(s2_wav_path)

        if data_dict["s1_video_path"] is not None:
            s1_video_path = data_dict["s1_video_path"]
            s1_video = self.load_video(s1_video_path)

            s2_video_path = data_dict["s2_video_path"]
            s2_video = self.load_video(s2_video_path)

        if data_dict["s1_embedding_path"] is not None:
            s1_embedding_path = data_dict["s1_embedding_path"]
            s1_embedding = self.load_object(s1_embedding_path)

            s2_embedding_path = data_dict["s2_embedding_path"]
            s2_embedding = self.load_object(s2_embedding_path)

        instance_data = {
            "mix": mix_audio,
            "s1": s1_audio,
            "s2": s2_audio,
            "s1_video": s1_video,
            "s2_video": s2_video,
            "s1_embedding": s1_embedding,
            "s2_embedding": s2_embedding,
            "audio_path": mix_wav_path,
        }
        # apply WAV augs before getting spec
        instance_data = self.preprocess_data(instance_data, single_key="mix")

        mix_spectrogram = self.get_spectrogram(mix_audio)
        instance_data.update({"mix_spectrogram": mix_spectrogram})

        s1_spectrogram = self.get_spectrogram(s1_audio)
        instance_data.update({"s1_spectrogram": s1_spectrogram})

        s2_spectrogram = self.get_spectrogram(s2_audio)
        instance_data.update({"s2_spectrogram": s2_spectrogram})

        if self.encoder:
            if hasattr(self.encoder, "stft"):
                complex_spectrogram = self.encoder.stft(mix_audio)
            else:
                raise AttributeError("Encoder should have stft method")
            instance_data.update({"complex_spectrogram": complex_spectrogram})

        # exclude WAV augs for prevending double augmentations
        instance_data = self.preprocess_data(
            instance_data, special_keys=["get_spectrogram", "mix"]
        )

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def load_video(self, path):
        data = np.load(path)
        return torch.FloatTensor(data["data"]).unsqueeze(0) #based on lipreader repo

    def get_spectrogram(self, audio):
        """
        Special instance transform with a special key to
        get spectrogram from audio.

        Args:
            audio (Tensor): original audio.
        Returns:
            spectrogram (Tensor): spectrogram for the audio.
        """
        return torch.log(self.instance_transforms["get_spectrogram"](audio).clamp(1e-5))

    def get_magnitude(self, audio):
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft),
            center=True,
            return_complex=True,
        )
        magnitude, phase = torch.abs(stft), torch.angle(stft)
        spec = 20.0 * torch.log10(torch.clamp(magnitude, min=1e-5)) - 20
        spec = torch.clamp(spec / 100, min=-1.0, max=0.0) + 1.0
        return spec, phase

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_object(self, path):
        """
        Load object from disk.

        Args:
            path (str): path to the object.
        Returns:
            data_object (Tensor):
        """
        if path.endswith('.npy'):
            data_object = torch.from_numpy(np.load(path))
        elif path.endswith('.npz'):
            with np.load(path) as data:
                data_object = torch.from_numpy(data[next(iter(data))])
        elif path.endswith('.pt') or path.endswith('.pth'):
            data_object = torch.load(path)
        
        return data_object.unsqueeze(0)

    def preprocess_data(
        self, instance_data, special_keys=["get_spectrogram"], single_key=None
    ):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
            single_key: optional[str]: if set modifies only this key
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is None:
            return instance_data

        if single_key is not None:
            if single_key in self.instance_transforms:  # eg train mode
                instance_data[single_key] = self.instance_transforms[single_key](
                    instance_data[single_key]
                )
            return instance_data

        for transform_name in self.instance_transforms.keys():
            if transform_name in special_keys:
                continue  # skip special key
            instance_data[transform_name] = self.instance_transforms[transform_name](
                instance_data[transform_name]
            )
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        some condition.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        # Filter logic
        pass

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "mix_wav_path" in entry, (
                "Each dataset item should include field 'mix_wav_path'"
                " - path to mix audio file."
            )

    @staticmethod
    def _sort_index(index):
        """
        Sort index via some rules.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting and after filtering.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        return sorted(index, key=lambda x: x["KEY_FOR_SORTING"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
