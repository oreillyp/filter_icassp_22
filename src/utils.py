import torch
import random
import os
import IPython.display as ipd
from IPython.core.display import display
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import PILToTensor, ToTensor
import pandas as pd

from typing import Union, List

################################################################################
# Utility functions
################################################################################


def tensor_to_np(x: torch.Tensor):
    """Convert tensor to NumPy array"""
    return x.clone().detach().cpu().numpy()


def set_random_seed(seed: int):
    """Set random seeds to allow for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(directory: Union[str, Path]):
    """Ensure all directories along given path exist, given directory name"""
    directory = str(directory)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)


def play_audio(x: torch.Tensor, sample_rate: int = 16000):
    display(ipd.Audio(tensor_to_np(x).flatten(), rate=sample_rate))


def plot_filter(amplitudes: torch.Tensor):
    """
    Given a single set of time-varying filter controls, return plot as image
    """

    amplitudes = amplitudes.clone().detach()

    if amplitudes.ndim == 2:
        magnitudes = amplitudes.cpu().numpy().T
    elif amplitudes.ndim == 3:
        magnitudes = amplitudes[0].cpu().numpy().T
    else:
        raise ValueError("Can only plot single filter response")

    # plot filter controls over time as heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(magnitudes, aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.invert_yaxis()
    ax.set_title('filter amplitudes')
    ax.set_xlabel('frames')
    ax.set_ylabel('frequency bin')
    plt.tight_layout()

    # save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # return plot as image
    return ToTensor()(np.array(img))


def plot_waveform(x: torch.Tensor, scale: Union[int, float] = 1.0):
    """Given single audio waveform, return plot as image"""
    try:
        assert len(x.shape) == 1 or x.shape[0] == 1
    except AssertionError:
        raise ValueError('Audio input must be single waveform')

    # waveform plot
    fig, ax = plt.subplots(figsize=(8,8))
    fig.subplots_adjust(bottom=0.2)
    plt.xticks(
        #rotation=90
    )
    ax.plot(tensor_to_np(x).flatten(), color='k')
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Waveform Amplitude")
    plt.axis((None, None, -scale, scale))  # set y-axis range

    # save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # return plot as image
    return ToTensor()(np.array(img))


def derange(x: torch.Tensor):
    """
    Shuffle tensor rows such that none remains in original position, using
    derangements with a fixed iteration budget; expected number of iterations
    required to shuffle with no fixed points is e (~3)
    """

    max_iter = 10
    orig_shape = x.shape
    x = x.reshape(x.shape[0], -1)

    rand_idx = torch.randperm(len(x))

    for i in range(max_iter):

        equal = torch.sum(
            1.0 * (x == x[rand_idx]),
            dim=-1
        ) >= x.shape[-1]

        if not equal.sum().item():
            break

        rand_idx = torch.randperm(len(x))

    return x[rand_idx].reshape(orig_shape)


def select_random_targets(y_true: torch.Tensor, n_per_class: int = None):
    """
    Given a batch of targets, permute such that no target is unchanged
    """

    y_new = y_true.clone()

    if n_per_class is not None and n_per_class > 1:
        assert not len(y_true) % n_per_class

        for i in range(n_per_class):
            # shuffle the ith strided subset
            idx_select = list(range(i, len(y_true) + i, n_per_class))
            while True:
                y_new[idx_select] = y_new[idx_select][torch.randperm(
                    y_new[idx_select].shape[0]
                )]
                if (y_new[idx_select] == y_true[idx_select]).sum() == 0:
                    break
    else:
        # inefficient
        while True:
            y_new = y_new[torch.randperm(y_new.shape[0])]
            if (y_new == y_true).sum() == 0:
                break
    return y_new


def stratified_sample(data: torch.utils.data.Dataset,
                      n_per_class: int = 1,
                      target: int = None,
                      exclude: Union[int, float, List] = None
                      ):
    """
    Sample a fixed number of input-target pairs per class and return as tensors.
    Assumes inputs and targets are stored in attributes `.tx` and `.ty`,
    respectively. If target class is provided, set all labels to target and
    excise inputs with ground truth label matching target
    """

    x, y, y_target = [], [], []

    rand_idx = torch.randperm(len(data))

    inputs_shuffled = data.tx[rand_idx]
    labels_shuffled = data.ty[rand_idx]

    labels = torch.unique(data.ty)

    for l in labels:

        if isinstance(exclude, int) and l == exclude:
            continue
        if isinstance(exclude, float) and l == exclude:
            continue
        if isinstance(exclude, List) and l in exclude:
            continue

        x_l = inputs_shuffled[labels_shuffled == l][:n_per_class]
        y_l = labels_shuffled[labels_shuffled == l][:n_per_class]

        if target is not None:
            if target == l:
                continue
            else:
                y_target_l = torch.full_like(y_l, target)
                y_target.append(y_target_l)

        x.append(x_l)
        y.append(y_l)

    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    if target is not None:
        y_target = torch.cat(y_target, dim=0)
        return x, y, y_target
    else:
        return x, y, None
