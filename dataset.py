import random
from typing import List, Tuple, Union

import numpy as np
import soundfile
import torch
import torch.utils.data as data
from boltons import fileutils

import hparams
from stft.stft import STFT


def spect_loader(path:str, trim_start:int, return_phase=False, num_samples=16000, crop=True) -> Union[torch.Tensor, 
                                                                                                      Tuple[torch.Tensor, torch.Tensor]]:
    y, _ = soundfile.read(path)

    if crop:
        y = y[trim_start: trim_start + num_samples]  # trim 'trim_start' from start and crop 1 sec
        y = np.hstack((y, np.zeros((num_samples - len(y)))))

        assert y.shape == (num_samples,)

    stft = STFT(hparams.N_FFT, hparams.HOP_LENGTH)
    y = torch.FloatTensor(y).unsqueeze(0)
    assert y.shape == (1, num_samples)
    spect, phase = stft.transform(y)

    if return_phase:
        return spect, phase
    
    return spect

def make_single_dataset(path, message_file, n_pairs):
    pairs = []
    wav_files = list(fileutils.iter_find_files(path, "*.wav"))

    for _ in range(n_pairs):
        sampled_file = random.sample(wav_files, 1)[0]
        pairs.append((sampled_file, message_file))

    return pairs

def make_pairs_dataset(path: str, n_pairs: int) -> List[Tuple[str, str]]:
    pairs = []
    wav_files = list(fileutils.iter_find_files(path, "*.wav"))

    for _ in range(n_pairs):
        sampled_files = random.sample(wav_files, 2)
        carrier_file, hidden_message_file = sampled_files
        pairs.append((carrier_file, hidden_message_file))
    return pairs


class TimitSingleDataset(data.Dataset):
    def __init__(self, root, message_file, n_pairs=10000,
                       trim_start=0, num_samples=16000):
       random.seed(0)
       self.spect_pairs = make_single_dataset(root, message_file, n_pairs)
       self.loader = spect_loader
       self.trim_start = int(trim_start)
       self.num_samples = num_samples

    def __getitem__(self, index):
        carrier_file, msg_file = self.spect_pairs[index]
        carrier_spect = self.loader(carrier_file, self.trim_start, num_samples=self.num_samples)
        msg_spect     = self.loader(msg_file, self.trim_start, num_samples=self.num_samples)
        
        return carrier_spect, msg_spect

    def __len__(self):
        return len(self.spect_pairs)


class TimitDataset(data.Dataset):
    def __init__(self, root,
                       n_pairs=100000,
                       transform=None,
                       trim_start=0,
                       num_samples=16000,
                       test=False):
        random.seed(0)
        self.spect_pairs = make_pairs_dataset(root, n_pairs)
        self.root = root
        self.transform = transform
        self.loader = spect_loader
        self.trim_start = int(trim_start)
        self.num_samples = num_samples
        self.test = test

    def __getitem__(self, index) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                          Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

        carrier_file, msg_file = self.spect_pairs[index]

        carrier_spect, carrier_phase = self.loader(carrier_file,
                                                   self.trim_start,
                                                   return_phase=True,
                                                   num_samples=self.num_samples)
        
        msg_spect, _ = self.loader(msg_file,
                                           self.trim_start,
                                           return_phase=True,
                                           num_samples=self.num_samples)

        if self.transform is not None:
            carrier_spect = self.transform(carrier_spect)
            carrier_phase= self.transform(carrier_phase)
            msg_spect = self.transform(msg_spect)

        return carrier_spect, msg_spect

    def __len__(self):
        return len(self.spect_pairs)
