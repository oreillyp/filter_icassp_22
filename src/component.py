import torch
import torch.nn as nn

import math

from src.constants import *

################################################################################
# Base class for differentiable signal-processing components
################################################################################


class Component(nn.Module):
    """
    Base class for differentiable audio-processing units
    """
    def __init__(self,
                 compute_grad: bool = True,
                 sample_rate: int = SR,
                 signal_length: int = SIG_LEN
                 ):
        super().__init__()
        self.compute_grad = compute_grad
        self.sample_rate = sample_rate

        # if signal length is given as floating-point value, assume time in
        # seconds and convert to samples
        if isinstance(signal_length, float):
            self.signal_length = math.floor(signal_length * self.sample_rate)
        else:
            self.signal_length = signal_length

    def forward(self, x: torch.Tensor):
        raise NotImplementedError()
