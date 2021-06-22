from typing import Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import math
import soundfile as sf

class sinc_Conv1d(nn.Module):
    @staticmethod
    def hz_to_mel(hz: int) -> int:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def mel_to_hz(mel: np.array) -> np.array:
        return 700 * (10 ** (mel / 2595) - 1)

    @staticmethod
    def g(n: torch.tensor,
          f1: torch.tensor,
          f2: torch.tensor) -> torch.tensor:
        '''
        g(n, f1, f2) = 2*f2*sinc(2π*f2*n) − 2*f1*sinc(2π*f1*n)

        '''
        f_times_t_low = torch.matmul(f1, n)
        f_times_t_high = torch.matmul(f2, n)
        return (torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(n/2)

    @staticmethod
    def compute_kernel(f1: torch.tensor,
                       f2: torch.tensor,
                       n: torch.tensor,
                       window: torch.tensor) -> torch.tensor:
        band = (f2 - f1)[:,0]

        band_pass_left = sinc_Conv1d.g(n, f1, f2)*window
        band_pass_center = 2*band.unsqueeze(-1)
        band_pass_right = torch.flip(band_pass_left,dims=[1])

        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)
        return band_pass / (2*band[:,None])
    
    def __init__(self, out_channels: int, kernel_size: int, sample_rate: int = 16000,
                 stride: int = 1, padding: int = 0, dilation: int =1):
        super(sinc_Conv1d, self).__init__()

        self.out_channels = out_channels
        assert kernel_size != 0, 'kernel size must be odd'
        self.kernel_size = kernel_size
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sample_rate = sample_rate

        # initialize filterbanks
        min_freq = 30
        self.min_f1 = 50
        self.min_band = 50
        max_freq = self.sample_rate / 2 - (self.min_f1 + self.min_band)

        freq_set_mel = np.linspace(self.hz_to_mel(min_freq),
                          self.hz_to_mel(max_freq),
                          self.out_channels + 1)
        freq_set = self.mel_to_hz(freq_set_mel)

        # filter lower frequency (out_channels, 1)
        self.f1 = nn.Parameter(torch.Tensor(freq_set[:-1]).unsqueeze(-1))

        # filter frequency band (out_channels, 1)
        self.band = nn.Parameter(torch.Tensor(np.diff(freq_set)).unsqueeze(-1))

        # Hamming window
        n_lin = torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2)))
        self.window = 0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);

        # (1, kernel_size/2)
        self.n = 2*math.pi*torch.arange(-(self.kernel_size - 1) / 2.0, 0).unsqueeze(0) / self.sample_rate

        
    
    def forward(self, inp: torch.tensor) -> torch.tensor:

        self.n = self.n.to(inp.device)
        self.window = self.window.to(inp.device)

        f1 = self.min_f1  + torch.abs(self.f1)
        
        f2 = torch.clamp(f1 + self.min_band + torch.abs(self.band), self.min_f1, self.sample_rate/2)        

        self.filters = (self.compute_kernel(f1,f2,self.n, self.window)).view(
            self.out_channels, 1, self.kernel_size)
        
        return F.conv1d(inp, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None) 


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # print((self.gamma * (x - mean) / (std + self.eps) + self.beta).dtype)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta




class SincNet(nn.Module):
    def __init__(self, input_dim: int, sample_rate:int = 16000, sinc_kernel: int = 251, 
                 sinc_filters: int = 80, cnn_filters: int = 60, cnn_kernel: int = 5,
                 maxpool_kernel: int = 3, hidden_dim: int = 2048, num_classes: int = 462):
        super(SincNet, self).__init__()

        self.input_dim = input_dim
        self.sample_rate = sample_rate

        self.sinc_kernel = sinc_kernel
        self.sinc_filters = sinc_filters
        self.cnn_filters = cnn_filters
        self.cnn_kernel = cnn_kernel
        self.maxpool_kernel = maxpool_kernel

        self.input_layer_norm = LayerNorm(self.input_dim)
        self.sinc = sinc_Conv1d(out_channels=self.sinc_filters, kernel_size=self.sinc_kernel, 
                                sample_rate=self.sample_rate)
        self.dim_cnn_1 = int((input_dim-self.sinc_kernel+1)/self.maxpool_kernel)
        self.dim_cnn_2 = int((self.dim_cnn_1-self.cnn_kernel+1)/self.maxpool_kernel)
        self.dim_cnn_3 = int((self.dim_cnn_2-self.cnn_kernel+1)/self.maxpool_kernel)
        self.dim_cnn_out = self.dim_cnn_3*self.cnn_filters
        self.sinc_layer_norm = LayerNorm(features=[self.sinc_filters,self.dim_cnn_1])

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            nn.Conv1d(self.sinc_filters, self.cnn_filters, self.cnn_kernel),
            nn.MaxPool1d(self.maxpool_kernel),
            LayerNorm(features=[self.cnn_filters, self.dim_cnn_2]),
            nn.LeakyReLU(),
            nn.Conv1d(self.cnn_filters, self.cnn_filters, self.cnn_kernel),
            nn.MaxPool1d(self.maxpool_kernel),
            LayerNorm(features=[self.cnn_filters, self.dim_cnn_3]),
            nn.LeakyReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_cnn_out, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim,momentum=0.05),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim,momentum=0.05),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim,momentum=0.05),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x = self.input_layer_norm(x)
        x = x.view(batch_size,1,seq_len)
        # sinc_Conv1d
        x = F.leaky_relu(self.sinc_layer_norm(F.max_pool1d(torch.abs(
                                            self.sinc(x)), kernel_size=3)))
        # Conv1d
        x = self.cnn(x).view(batch_size,-1)
        # MLP
        x = self.mlp(x)
        return x
