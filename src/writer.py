import torch
import logging
import sys
import time
from datetime import datetime
import os
import librosa as li
import glob
from os import path
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import PILToTensor, ToTensor
import io
import numpy as np

from src.utils import plot_waveform, plot_filter, tensor_to_np
from src.constants import *

################################################################################
# Utility functions
################################################################################


class Writer:
    """
    Encapsulates file, console, and TensorBoard logging utilities
    """
    def __init__(self,
                 root_dir: str = RUNS_DIR,
                 name: str = None,
                 use_tb: bool = False,
                 tb_log_iter: int = 100,
                 ):
        """
        Configure logging.

        :param root_dir: root logging directory
        :param name: descriptive name for run
        :param use_tb: if True, use TensorBoard
        :param tb_log_iter: if `use_tb` is True
        """

        # generate run-specific name and create directory
        run_name = f'{name}_{time.strftime("%m-%d-%H:%M:%S")}'
        self.run_dir = Path(root_dir) / run_name
        ensure_dir(self.run_dir)

        # log to TensorBoard
        self.use_tb = use_tb
        self.tb_log_iter = tb_log_iter
        self.writer = SummaryWriter(
            log_dir=str(self.run_dir),
            flush_secs=20,
        ) if use_tb else None

        # log to console and file
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(
            logging.StreamHandler(sys.stdout)
        )
        self.logger.addHandler(
            logging.FileHandler(self.run_dir / 'log.txt')
        )

        self.logger.info(f'Logging to {self.run_dir}')

        # disable Matplotlib logging
        logging.getLogger('matplotlib.font_manager').disabled = True

    def log_info(self, info: str):
        self.logger.info(info)

    def log_loss(self, x: torch.Tensor, tag: str, global_step: int = 0):
        """
        Log scalar loss
        """

        if not self.tb_log_iter or global_step % self.tb_log_iter:
            return

        # log scalar to file and console
        self.logger.info(f'iter {global_step}\tloss/{tag}: {x}')

        # log plot to TensorBoard
        if self.writer:
            self.writer.add_scalar(f'{tag}', x, global_step=global_step)
            self.writer.flush()

    def log_gradient(self, x: torch.Tensor, tag: str, global_step: int = 0):
        """
        Plot gradients for single input
        """

        if not self.tb_log_iter or global_step % self.tb_log_iter:
            return

        if self.writer:

            norm = torch.norm(x, p=2)

            # scalar plot: norm of gradient vector
            self.writer.add_scalar(f'{tag}-norm', norm, global_step=global_step)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.tick_params(color='#159957', labelcolor='#606c71')
            for spine in ax.spines.values():
                spine.set_edgecolor('#dce6f0')

            ax.plot(tensor_to_np(x).flatten(), linewidth=1, color="#159957")
            ax.grid(True)

            # set labels
            plt.setp(ax, xlabel='Sample')
            plt.setp(ax, ylabel='Amplitude')

            plt.axis((None, None, -1.0, 1.0))  # set y-axis range

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)
            self.writer.add_image(f"{tag}", ToTensor()(np.array(img)), global_step=global_step)
            self.writer.flush()

    def log_filter(self, amplitudes: torch.Tensor, tag: str, global_step: int = 0):
        """
        Plot filter controls
        """

        if not self.tb_log_iter or global_step % self.tb_log_iter:
            return
        if self.writer:
            if amplitudes.ndim == 2:
                magnitudes = amplitudes.cpu().numpy().T
            elif amplitudes.ndim == 3:
                magnitudes = amplitudes[0].cpu().numpy().T
            else:
                raise ValueError("Can only plot single filter response")

            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(magnitudes, aspect='auto')
            fig.colorbar(im, ax=ax)
            ax.invert_yaxis()
            ax.set_title('filter amplitudes')
            ax.set_xlabel('frames')
            ax.set_ylabel('frequency bin')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)
            self.writer.add_image(
                f'filter_controls/{tag}',
                ToTensor()(np.array(img)),
                global_step
            )
            self.writer.flush()

    def log_audio(self,
                  x: torch.Tensor,
                  tag: str,
                  global_step: int = 0,
                  sample_rate: int = SR):
        """
        Log waveform plot, audio file, and spectrogram
        """

        if not self.tb_log_iter or global_step % self.tb_log_iter:
            return

        if self.writer:

            x = x.clone().detach().cpu()

            try:
                assert len(x.shape) == 1 or x.shape[0] == 1
            except AssertionError:
                raise ValueError('Audio logging input must be single waveform')

            # audio playback
            sr = sample_rate
            scale = 1.0

            normalized = (scale / torch.max(  # avoid clipping, normalize volume
                torch.abs(x) + 1e-12, dim=-1, keepdim=True)[0]) * x * 0.95

            self.writer.add_audio(f"{tag}",
                                  normalized,
                                  sample_rate=sr,
                                  global_step=global_step)

            # waveform plot
            fig, ax = plt.subplots(figsize=(8,8))
            fig.subplots_adjust(bottom=0.2)
            plt.xticks(
                #rotation=90
            )
            ax.plot(tensor_to_np(x).flatten(), color='k')
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Waveform Amplitude")
            plt.axis((None, None, -1.0, 1.0))  # set y-axis range

            # add iteration number as text
            plt.text(3 * x.shape[-1] // 8, 0.80, f'Step: {global_step}', fontsize=18)

            #fig = plt.figure(figsize=(8, 8))
            #plt.plot(tensor_to_np(x).flatten())
            #plt.axis((None, None, -1.0, 1.0))  # set y-axis range

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)
            self.writer.add_image(f"{tag}-waveform",
                                  ToTensor()(np.array(img)),
                                  global_step=global_step)

            # spectrogram plot
            spec = torch.stft(x.reshape(1, -1),
                              n_fft=512,
                              win_length=512,
                              hop_length=256,
                              window=torch.hann_window(window_length=512),
                              return_complex=True,
                              center=False
                              )
            spec = torch.squeeze(torch.abs(spec) / (torch.max(torch.abs(spec)))) # normalize by maximum absolute value

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pcolormesh(tensor_to_np(torch.log(spec + 1)), vmin=0, vmax=.31)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)
            self.writer.add_image(f"{tag}-spectrogram", ToTensor()(np.array(img)), global_step)

            self.writer.flush()