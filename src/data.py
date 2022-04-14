from torch.utils.data import Dataset

import math
import glob
import os
from os import path
from tqdm import tqdm
import librosa as li

from src.constants import *

from typing import Union

################################################################################
# Preprocess data and cache to disk in tensor format
################################################################################


class LibriSpeechDataset(Dataset):
    """
    A Dataset object for the LibriSpeech dataset subsets. The required data can
    be downloaded by running the script `download_librispeech.sh`. This class
    takes audio data from the specified directory and caches tensors to disk.
    """
    def __init__(self,
                 split: str = 'test-clean',
                 data_dir: str = DATA_DIR / 'LibriSpeech',
                 cache_dir: str = CACHE_DIR / 'LibriSpeech',
                 sample_rate: int = SR,
                 ext: str = LIBRISPEECH_EXT,
                 signal_length: Union[float, int] = SIG_LEN,
                 **kwargs):
        """
        Load, organize, and cache Speech Commands dataset.

        :param split: named subset
        :param data_dir: LibriSpeech root directory
        :param cache_dir: directory to which tensors will be saved
        :param sample_rate: sample rate
        :param ext: extension for audio files within dataset
        :param signal_length: length of audio files in samples (if `int` given)
                              or seconds (if `float` given)
        """

        self.data_dir = os.fspath(data_dir)
        self.cache_dir = os.fspath(cache_dir)

        if split not in [
            'test-clean',
            'test-other',
            'dev-clean',
            'dev-other',
            'train-clean-100',
            'train-clean-360',
            'train-other-500'
        ]:
            raise ValueError(f'Invalid split {split}')

        self.split = split

        # create directories if necessary
        ensure_dir(path.join(self.cache_dir, self.split))
        ensure_dir(path.join(self.cache_dir, self.split))

        self.ext = ext
        self.sample_rate = sample_rate

        # if signal length is given as floating-point value, assume time in
        # seconds and convert to samples
        if isinstance(signal_length, float):
            self.signal_length = math.floor(signal_length * self.sample_rate)
        else:
            self.signal_length = signal_length

        # check for cached tensors and metadata; if present, load and exit
        cache_tensors = list(
            (Path(self.cache_dir) / self.split).rglob('*.pt')
        )

        if len(cache_tensors) >= 2:
            self.tx = torch.load(
                path.join(
                    self.cache_dir, self.split, 'tx.pt'
                )
            )
            self.ty = torch.load(
                path.join(
                    self.cache_dir, self.split, 'ty.pt'
                )
            )

        else:
            self.tx, self.ty = self._build_cache()

        self.n_classes = torch.unique(self.ty).shape[-1]
        self.classes = [c.item() for c in torch.unique(self.ty)]

        # trim or pad audio to signal length as necessary
        if self.tx.shape[-1] > self.signal_length:
            self.tx = self.tx[..., :self.signal_length]
        elif self.tx.shape[-1] < self.signal_length:
            self.tx = F.pad(
                self.tx, (0, self.signal_length - self.tx.shape[-1])
            )

    def __str__(self):
        """Return string representation of dataset"""
        return f'LibriSpeechDataset(split={self.split})'

    def _build_cache(self):
        """
        Process and cache data
        :return: data tensor and targets tensor
        """
        # enumerate speaker IDs (subdirectories within split)
        spkr_ids = glob.glob(
            f'{str(Path(self.data_dir) / self.split)}/*/'
        )
        spkr_ids = [int(Path(i).name) for i in spkr_ids]

        # catalog audio
        audio_list = sorted(
            list(
                (Path(self.data_dir) / self.split).rglob(f'*.{self.ext}')
            )
        )

        # cache dataset in tensor form
        tx = torch.zeros(
            (len(audio_list), 1, self.signal_length)
        )
        ty = torch.zeros(
            len(audio_list), dtype=torch.long
        )

        pbar = tqdm(audio_list, total=len(audio_list))
        for i, audio_fn in enumerate(pbar):
            pbar.set_description(
                f'Loading LibriSpeech ({self.split}): '
                f'{path.basename(audio_fn)}')

            # extract speaker ID and load waveform audio
            label = int(Path(audio_fn).parts[-3])
            waveform, _ = li.load(audio_fn,
                                  mono=True,
                                  sr=self.sample_rate,
                                  duration=self.signal_length // self.sample_rate)
            waveform = torch.from_numpy(waveform)[..., :self.signal_length]

            tx[i, :, :waveform.shape[-1]] = waveform
            ty[i] = label

        # cache tensors
        torch.save(tx, path.join(self.cache_dir, self.split, 'tx.pt'))
        torch.save(ty, path.join(self.cache_dir, self.split, 'ty.pt'))

        # return tensors for requested split alongside data dictionary
        return tx, ty

    def __len__(self):
        return self.tx.shape[0]

    def __getitem__(self, idx):
        return self.tx[idx], self.ty[idx]
