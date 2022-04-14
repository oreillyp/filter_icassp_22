import torch
import math
import decimal

from src.component import Component

################################################################################
# Voice activity detection
################################################################################


class VAD(Component):
    """
    Apply Voice Activity Detection (VAD) while allowing for straight-through
    gradient estimation. For now, only supports simple energy-based method,
    and should be placed after normalization to avoid scale-dependence.
    """
    def __init__(self,
                 compute_grad: bool = True,
                 frame_len: float = 0.05,
                 threshold: float = -72
                 ):

        super().__init__(compute_grad)

        self.threshold = threshold
        self.frame_len = int(
            decimal.Decimal(
                frame_len * self.sample_rate
            ).quantize(
                decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP
            )
        )  # convert seconds to samples, round up

    def forward(self, x: torch.Tensor):

        # require batch dimension
        assert x.ndim >= 2

        # require mono audio, discard channel dimension
        n_batch, slen = x.shape[0], x.shape[-1]
        audio = x.reshape(n_batch, slen)

        eps = 1e-12  # numerical stability

        # determine number of frames
        if slen <= self.frame_len:
            n_frames = 1
        else:
            n_frames = 1 + int(
                math.ceil(
                    (1.0 * slen - self.frame_len) / self.frame_len)
            )

        # pad to integer frame length
        padlen = int(n_frames * self.frame_len)
        zeros = torch.zeros((x.shape[0], padlen - slen,)).to(x)
        padded = torch.cat((audio, zeros), dim=-1)

        # obtain strided (frame-wise) view of audio
        shape = (padded.shape[0], n_frames, self.frame_len)
        frames = torch.as_strided(
            padded,
            size=shape,
            stride=(padded.shape[-1], self.frame_len, 1)
        )

        # create frame-by-frame mask based on energy threshold
        mask = 20 * torch.log10(
            ((frames).norm(dim=-1) / self.frame_len) + eps
        ) > self.threshold

        # turn frame-by-frame mask into sample-by-sample mask
        mask_wav = torch.repeat_interleave(mask, self.frame_len, dim=-1)
        samples_per_row = torch.sum(mask, dim=-1) * self.frame_len

        split = torch.split(padded[mask_wav], tuple(samples_per_row))

        # placeholder for outputs: (n_batch, 1, padded_length)
        final = torch.zeros_like(padded).unsqueeze(1)  # pad to preserve length

        # concatenate and pad split views
        for i, tensor in enumerate(split):
            length = tensor.shape[-1]
            final[i, :, :length] = tensor

        return final[..., :slen]
