import random
from typing import List, Tuple, Union

import numpy as np
import soundfile
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from boltons import fileutils
from hparams import AUDIO_LEN


import hparams
from stft.stft import STFT

from config import DEBUG, debug_print



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

    # # Magnitude Spectrum (振幅谱)
    # magnitude = spect.abs()
    
    # # Phase Spectrum (相位谱)
    # phase = torch.angle(spect)

    # # Power Spectrum (功率谱)
    # power = magnitude ** 2
    if return_phase:
        return spect, phase
    
    return spect

# def spect_loader_three_channels(path:str, trim_start:int, num_samples=16000, crop=True) -> torch.Tensor:
#     y, _ = soundfile.read(path)

#     if crop:
#         y = y[trim_start: trim_start + num_samples]
#         y = np.hstack((y, np.zeros((num_samples - len(y)))))  # Pad with zeros if necessary

#     stft = STFT(hparams.N_FFT, hparams.HOP_LENGTH)
#     y = torch.FloatTensor(y).unsqueeze(0)  # Convert to Tensor and add batch dimension
#     spect, phase = stft.transform(y)  # Get the magnitude and phase

#     # Magnitude Spectrum (振幅谱)
#     magnitude = spect.abs()
    
#     # Phase Spectrum (相位谱)
#     phase = torch.angle(spect)

#     # Power Spectrum (功率谱)
#     power = magnitude ** 2

#     # Combine the features as channels: magnitude, phase, and power
#     # You can also apply a log transformation to the power and magnitude if needed
#     combined = torch.cat((magnitude, phase, power), dim=1)

#     return combined
# def spect_loader_three_channels(path: str, trim_start: int, num_samples=16000, crop=True) -> torch.Tensor:
#     """
#     加载音频并返回三通道频谱图：振幅、相位、功率
#     """
#     y, _ = soundfile.read(path)

#     if crop:
#         y = y[trim_start: trim_start + num_samples]
#         y = np.hstack((y, np.zeros((num_samples - len(y)))))  # Pad with zeros if necessary

#     stft = STFT(hparams.N_FFT, hparams.HOP_LENGTH)
#     y = torch.FloatTensor(y).unsqueeze(0)  # Convert to Tensor and add batch dimension
    
#     # 获取复数频谱
#     complex_spect, _ = stft.transform(y)  # complex_spect 是复数张量
    
#     # 打印调试信息
#     debug_print(f"Audio file: {path}")
#     debug_print(f"Input audio shape: {y.shape}")
#     debug_print(f"Complex spectrogram shape: {complex_spect.shape}")
#     debug_print(f"Complex spectrogram dtype: {complex_spect.dtype}")
    
#     # 从复数频谱计算三个特征
#     # 1. Magnitude Spectrum (振幅谱)
#     magnitude = torch.abs(complex_spect)
    
#     # 2. Phase Spectrum (相位谱)
#     phase = torch.angle(complex_spect)
    
#     # 3. Power Spectrum (功率谱)
#     power = magnitude ** 2
    
#     debug_print(f"Magnitude shape: {magnitude.shape}")
#     debug_print(f"Phase shape: {phase.shape}")
#     debug_print(f"Power shape: {power.shape}")
    
#     # 使用 torch.stack 在新维度上堆叠三个通道
#     # 输入：三个 [1, 129, 378] 的张量
#     # 输出：[3, 129, 378]，然后添加batch维度得到 [1, 3, 129, 378]
#     combined = torch.stack([magnitude.squeeze(0), phase.squeeze(0), power.squeeze(0)], dim=0)
#     combined = combined.unsqueeze(0)  # 添加batch维度
    
#     debug_print(f"Final output shape: {combined.shape}")
    
#     return combined

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

class TimitSingleDatasetWithThreeChannels(data.Dataset):
    def __init__(self, root, message_file, n_pairs=10000,
                       trim_start=0, num_samples=16000):
       random.seed(0)
       self.spect_pairs = make_single_dataset(root, message_file, n_pairs)
       self.loader = spect_loader_three_channels  # 使用新的loader
       self.trim_start = int(trim_start)
       self.num_samples = num_samples

    def __getitem__(self, index):
        carrier_file, msg_file = self.spect_pairs[index]
        
        # 获取载体信号和消息信号的三个通道频谱图 振幅 相位 功率
        carrier_spect = self.loader(carrier_file, self.trim_start, num_samples=self.num_samples)
        msg_spect     = self.loader(msg_file, self.trim_start, num_samples=self.num_samples)
        debug_print(f"Dataset __getitem__ - Carrier spect shape: {carrier_spect.shape}")  
        debug_print(f"Dataset __getitem__ - Message spect shape: {msg_spect.shape}")    
        
        # 确保形状正确：应该是 [1, 3, height, width]，我们需要去掉batch维度
        if carrier_spect.ndim == 4 and carrier_spect.shape[0] == 1:
            carrier_spect = carrier_spect.squeeze(0)  # [3, height, width]
        if msg_spect.ndim == 4 and msg_spect.shape[0] == 1:
            msg_spect = msg_spect.squeeze(0)  # [3, height, width]
            
        debug_print(f"Dataset __getitem__ - Final carrier spect shape: {carrier_spect.shape}")  
        debug_print(f"Dataset __getitem__ - Final message spect shape: {msg_spect.shape}")
        
        return carrier_spect, msg_spect

    def __len__(self):
        return len(self.spect_pairs)


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



def train_dataloader(train_path, batch_size, num_workers):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    train_dataset = TimitDataset(train_path,
                                n_pairs     = 4608,
                                trim_start  = trim_start,
                                num_samples = num_samples)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size  = batch_size,
                                  shuffle     = True,
                                  num_workers = num_workers)
    
    return train_dataloader

def val_dataloader(val_path, batch_size, num_workers):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    val_dataset = TimitDataset(val_path,
                               n_pairs     = 832,
                               trim_start  = trim_start,
                               num_samples = num_samples,
                               test        = True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size  = batch_size,
                                shuffle     = False,
                                num_workers = num_workers)
    
    return val_dataloader

def test_dataloader(test_path, batch_size):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    test_dataset = TimitDataset(test_path,
                                n_pairs     = 832,
                                trim_start  = trim_start,
                                num_samples = num_samples,
                                test        = True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size  = batch_size,
                                 shuffle     = False,
                                 num_workers = 0)
    
    return test_dataloader

def unet_train_single_dataloader(single_path, message_file, batch_size, num_workers):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    single_dataset = TimitSingleDataset(single_path,
                                 message_file,
                                n_pairs     = 4608,
                                # n_pairs=32,
                                trim_start  = trim_start,
                                num_samples = num_samples)
    single_dataloader = DataLoader(single_dataset,
                                  batch_size  = batch_size,
                                  shuffle     = True,
                                  num_workers = num_workers)
    
    return single_dataloader

def train_single_dataloader(train_path, message_file, batch_size, num_workers):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    train_dataset = TimitSingleDataset(train_path,
                                 message_file,
                                n_pairs     = 4608,
                                # n_pairs=32,
                                trim_start  = trim_start,
                                num_samples = num_samples)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size  = batch_size,
                                  shuffle     = True,
                                  num_workers = num_workers)
    
    return train_dataloader

def unet_val_single_dataloader(val_path, message_file, batch_size, num_workers):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    val_dataset = TimitSingleDataset(val_path,
                                     message_file,
                               n_pairs     = 832,
                               # n_pairs=32,
                               trim_start  = trim_start,
                               num_samples = num_samples)
    val_dataloader = DataLoader(val_dataset,
                                batch_size  = batch_size,
                                shuffle     = False,
                                num_workers = num_workers)
    
    return val_dataloader

def val_single_dataloader(val_path, message_file, batch_size, num_workers):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    val_dataset = TimitSingleDataset(val_path,
                                     message_file,
                               n_pairs     = 832,
                               # n_pairs=32,
                               trim_start  = trim_start,
                               num_samples = num_samples)
    val_dataloader = DataLoader(val_dataset,
                                batch_size  = batch_size,
                                shuffle     = False,
                                num_workers = num_workers)
    
    return val_dataloader

def test_single_dataloader(test_path, message_file, batch_size):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    test_dataset = TimitDataset(test_path,
                                message_file,
                                n_pairs     = 832,
                                # n_pairs=32,
                                trim_start  = trim_start,
                                num_samples = num_samples)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size  = batch_size,
                                 shuffle     = False,
                                 num_workers = 0)
    
    return test_dataloader
