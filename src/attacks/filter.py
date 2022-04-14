import torch
import torch.nn as nn

import numpy as np

import torch.fft as fft
import math

from tqdm import tqdm

from src.pipelines.pipeline import Pipeline
from src.writer import Writer
from src.loss.adversarial import AdversarialLoss
from src.loss.auxiliary import AuxiliaryLoss
from src.loss.speaker_embedding import SpeakerEmbeddingLoss

from typing import Optional, Union, Tuple, Iterable, TYPE_CHECKING

################################################################################
# Time-varying filter attack
################################################################################


class Filter(nn.Module):
    """
    Apply a time-varying (adaptive) FIR filter to benign waveform audio via
    FFT convolution
    """
    def __init__(self,
                 x: torch.Tensor,
                 block_size: int,
                 n_bands: int,
                 win_type: str = 'hann',
                 normalize_ir: Union[str, int] = None,
                 normalize_audio: str = 'peak'
                 ):
        """
        Store frame length and per-frame filter controls in the form of
        frequency magnitude responses over a fixed number of bands

        :param x: reference audio tensor, shape (n_batch, signal_len)
        :param block_size: frame length in samples
        :param n_bands: number of filter bands
        """
        super().__init__()

        self.block_size = block_size
        self.n_bands = n_bands
        self.win_type = win_type
        self.normalize_ir = normalize_ir
        self.normalize_audio = normalize_audio

        if self.win_type not in ['rectangular', 'triangular', 'hann']:
            raise ValueError(f'Invalid window type {win_type}')

        if self.normalize_ir not in [None, 'none', 1, 2]:
            raise ValueError(f'Invalid IR normalization type {normalize_ir}')

        if self.normalize_audio not in [None, 'none', 'peak']:
            raise ValueError(f'Invalid audio normalization type {normalize_audio}')

        assert x.ndim > 1  # require batch dimension
        n_batch, signal_len = x.shape[0], x.shape[-1]

        if self.win_type == 'rectangular':  # non-overlapping frames
            n_frames = signal_len // self.block_size + 1
        elif self.win_type in ['triangular', 'hann']:  # 50% overlap
            hop_size = block_size // 2
            n_frames = signal_len // hop_size + 1

        # start with filter gains at unity
        self.amplitudes = nn.Parameter(
            torch.ones(n_batch, n_frames, n_bands).float().to(x.device)
        )

    @staticmethod
    def _get_win_func(win_type: str):
        if win_type == 'rectangular':
            return lambda m: torch.ones(m)
        elif win_type == 'hann':
            return lambda m: torch.as_tensor(np.hanning(m))
        elif win_type == 'triangular':
            return lambda m: torch.as_tensor(np.bartlett(m))

    @staticmethod
    def _scale(x: torch.Tensor):
        """
        Scale control signal (frequency amplitude response) to fixed range
        """
        return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7

    def _fft_convolve(self, signal: torch.Tensor, kernel: torch.Tensor, n_fft: int):
        """
        Given waveform representations of signal and FIR filter set, convolve
        via point-wise multiplication in Fourier domain
        """

        # right-pad kernel and frames to n_fft samples
        signal = nn.functional.pad(signal, (0, n_fft - signal.shape[-1]))
        kernel = nn.functional.pad(kernel, (0, n_fft - kernel.shape[-1]))

        convolved = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))

        # account for prior shift to symmetric (zero-phase) form
        rolled = torch.roll(convolved, shifts=-(self.n_bands - 1), dims=-1)

        return rolled

    def _amp_to_ir(self, amp: torch.Tensor):
        """
        Convert filter frequency amplitude response into a time-domain impulse
        response. The filter response is given as a per-frame transfer function,
        and a symmetric impulse response is returned

        :param amp: shape (n_batch, n_frames, n_bands) or (1, n_frames, n_bands)
        """

        # convert to complex zero-phase representation
        amp = torch.stack([amp, torch.zeros_like(amp)], -1)
        amp = torch.view_as_complex(amp)  # shape (n_batch, n_frames, n_bands)

        # compute 1D inverse FFT along final dimension, treating bands as
        # Fourier frequencies of analysis
        impulse = fft.irfft(amp)

        # require filter size to match time-domain transform of filter bands
        filter_size = impulse.shape[-1]

        # apply window to shifted zero-phase (symmetric) form of impulse
        impulse = torch.roll(impulse, filter_size // 2, -1)
        win = torch.hann_window(filter_size, dtype=impulse.dtype, device=impulse.device)

        if self.normalize_ir in [None, 'none']:
            pass
        elif self.normalize_ir == 1:
            impulse = impulse / torch.sum(impulse, dim=-1, keepdim=True)
        elif self.normalize_ir == 2:
            impulse = impulse / torch.norm(impulse, p=2, dim=-1, keepdim=True) + 1e-20

        return impulse * win  # TODO: should Hann window be applied to IR?

    def forward(self, x: torch.Tensor):
        """
        Apply time-varying FIR to audio input frame-by-frame using stored filter
        controls
        :param x: (n_batch, signal_len)
        """

        assert x.ndim > 1  # require batch dimension

        orig_shape = x.shape
        n_batch, signal_len = orig_shape[0], orig_shape[-1]
        x = x.reshape(n_batch, signal_len)  # assume mono audio input

        _, n_frames, n_bands = self.amplitudes.shape

        peak = torch.max(torch.abs(x), -1)[0].reshape(-1)

        # scale stored amplitudes to fixed range [0, 20]
        magnitudes = self._scale(self.amplitudes) * 10

        # convert filter controls (frequency amplitude responses) to frame-by-
        # frame time-domain impulse responses
        impulse = self._amp_to_ir(magnitudes)

        # using stored block size, pad or trim signal to match number of frames
        # in filter controls
        if self.win_type == 'rectangular':
            pad_len = n_frames * self.block_size
        elif self.win_type in ['triangular', 'hann']:
            pad_len = (n_frames - 1) * (self.block_size // 2) + self.block_size
        else:
            raise ValueError(f'Invalid window type {self.win_type}')
        if signal_len < pad_len:
            x = nn.functional.pad(x, (0, pad_len - signal_len))
        else:
            x = x[..., :pad_len]

        if self.win_type == 'rectangular':
            # frame padded audio with non-overlapping rectangular window
            x = x.unfold(-1, self.block_size, self.block_size)
        elif self.win_type in ['triangular', 'hann']:
            # use 50% overlap for COLA reconstruction
            x = x.unfold(-1, self.block_size, self.block_size // 2)

        # determine FFT size using inferred FIR waveform filter length
        n_fft_min = self.block_size + 2 * (n_bands - 1)
        n_fft = pow(2, math.ceil(math.log2(n_fft_min)))  # use next power of 2

        # convolve frame-by-frame in FFT domain; resulting padded frames will
        # contain "ringing" overlapping segments which must be summed
        signal = self._fft_convolve(x, impulse, n_fft).contiguous()  # shape (n_batch, n_frames, n_fft)

        # restore signal from frames using overlap-add
        if self.win_type == 'rectangular':
            pad_len = self.block_size * (n_frames - 1) + n_fft
            signal = nn.functional.fold(
                signal.permute(0, 2, 1),
                (1, pad_len),
                kernel_size=(1, n_fft),
                stride=(1, self.block_size)
            ).reshape(n_batch, -1)

        elif self.win_type in ['triangular', 'hann']:

            truncated_len = (
                                (
                                        (pad_len - self.block_size) // (self.block_size // 2)
                                )
                            ) * (self.block_size // 2) + n_fft
            win = self._get_win_func(self.win_type)(self.block_size).to(signal).reshape(1, 1, -1)
            win_pad_len = signal.shape[-1] - win.shape[-1]
            win = nn.functional.pad(win, (0, win_pad_len))

            signal = signal * win  # apply windowing

            # use `nn.functional.fold` to perform overlap-add synthesis; for
            # reference, see https://tinyurl.com/pw7mv9hh
            signal = nn.functional.fold(
                signal.permute(0, 2, 1),
                (1, truncated_len),
                kernel_size=(1, n_fft),
                stride=(1, self.block_size // 2)
            ).reshape(n_batch, -1)

        trimmed = signal[..., :signal_len].reshape(orig_shape)  # trim signal to original length

        if self.normalize_audio in [None, 'none']:
            factor = 1.0
        elif self.normalize_audio == 'peak':
            factor = peak / torch.max(torch.abs(trimmed), -1)[0].reshape(-1)
            factor = factor.reshape(-1, 1, 1)
        else:
            raise ValueError(f'Invalid audio normalization type {self.normalize_audio}')

        normalized = trimmed * factor

        return normalized


class FilterAttack:

    def __init__(self,
                 block_size: int = 512,
                 n_bands: int = 64,
                 pipeline: Pipeline = None,
                 class_loss: AdversarialLoss = None,
                 aux_loss: AuxiliaryLoss = None,
                 max_iter: int = 100,
                 eot_iter: int = 0,
                 opt: str = 'sgd',
                 lr: float = 0.1,
                 eps: Union[int, float, torch.Tensor] = 0.1,
                 batch_size: int = 32,
                 mode: str = None,
                 projection_norm: Union[int, float, str] = float('inf'),
                 k: int = None,
                 rand_evals: int = 0,
                 writer: Writer = None
                 ):
        """
        Adapted from `Evading Adversarial Example Detection Defenses with
        Orthogonal Projected Gradient Descent` by Bryniarski et al.
        (https://github.com/v-wangg/OrthogonalPGD).

        :block_size: frame length of time-varying filter controls
        :param pipeline: a Pipeline object wrapping a defended classifier
        :param class_loss: a classification loss object; must take model
                           predictions and targets as arguments
        :param aux_loss: an auxiliary loss object; must take original and
                         adversarial inputs as arguments
        :param max_iter: the maximum number of iterations for optimization
        :param eot_iter: resampling interval for pipeline simulation parameters;
                         if 0 or None, do not resample parameters
        :param opt: must be one of 'sgd', 'adam', or 'lbfgs'
        :param lr: learning rate for optimization
        :param eps: magnitude bound for adversarial perturbation
        :param mode: one of None, 'orthogonal', 'selective'
        :param projection_norm: gradient regularization, one of 'linf' or 'l2'
        :param k: if not None, perform gradient projection every kth step
        :param writer: a TensorBoard SummaryWriter for logging attack progress
        """

        self.block_size = block_size
        self.n_bands = n_bands
        self.pipeline = pipeline
        self.class_loss = class_loss
        self.aux_loss = aux_loss
        self.max_iter = max_iter
        self.eot_iter = eot_iter
        self.opt = opt
        self.lr = lr
        self.eps = eps
        self.batch_size = batch_size
        self.mode = mode
        self.projection_norm = projection_norm
        self.k = k
        self.rand_evals = rand_evals
        self.writer = writer

        self._batch_id = 0
        self._i_max_iter = 0

        self._check_loss(class_loss, aux_loss)

    @staticmethod
    def _check_loss(class_loss: AdversarialLoss, aux_loss: AuxiliaryLoss):
        try:
            assert class_loss is not None
        except AssertionError:
            raise ValueError('Must provide classification loss')

        try:
            assert class_loss.reduction is None or class_loss.reduction == 'none'
        except AssertionError:
            raise ValueError('All losses must provide unreduced scores')

        if aux_loss is not None:
            try:
                assert aux_loss.reduction is None or aux_loss.reduction == 'none'
            except AssertionError:
                raise ValueError('All losses must provide unreduced scores')

    @staticmethod
    def _dot(x1: torch.Tensor, x2: torch.Tensor):
        """
        Compute batch dot product between tensors along last dimension
        """
        return (x1*x2).sum(-1, keepdim=True)

    @staticmethod
    def _create_dataset(x: torch.Tensor, y: torch.Tensor):

        dataset = torch.utils.data.TensorDataset(
            x.to(torch.float32),
            y.to(torch.float32),
        )

        return dataset

    def _compute_success_array(self,
                               x_clean: torch.Tensor,
                               labels: torch.Tensor,
                               x_adv: torch.Tensor
                               ):
        """
        Evaluate success of attack batch
        """
        preds = self.pipeline(x_clean.detach())
        adv_preds = self.pipeline(x_adv.detach())

        if self.class_loss.targeted:
            attack_success = self.pipeline.match_predict(adv_preds, labels)
        else:
            attack_success = ~self.pipeline.match_predict(adv_preds, preds)

        return attack_success

    def _project_orthogonal(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Compute projection component of x1 along x2. For projection
        onto zero vector, return zero vector
        """
        return x2 * (self._dot(x1, x2) / self._dot(x2, x2).clamp_min(1e-12))

    def _component_orthogonal(self,
                              x1: torch.Tensor,
                              x2: torch.Tensor):
        """
        Compute component of x1 approximately orthogonal to x2
        """
        return x1 - self._project_orthogonal(x1, x2)

    def _check_compatibility_input_and_eps(self, x: torch.Tensor):
        """
        Check the compatibility of the input with projection bound `eps`. Bound
        must be scalar or a batch-length tensor
        :param x: inputs of shape (n_examples, ...)
        """
        if self.eps is None:
            return
        elif isinstance(self.eps, torch.Tensor):
            if len(self.eps.flatten()) == 1:
                pass
            else:
                n_batch = x.shape[0]
                if len(torch.flatten(self.eps)) != n_batch:
                    raise ValueError(
                        "Epsilon must be scalar (float) or batch-length tensor"
                    )
                self.eps = self.eps.reshape(n_batch, 1)
        elif isinstance(self.eps, float):
            self.eps = torch.as_tensor(
                self.eps
            ).to(torch.float32).reshape(1, 1)
        else:
            raise ValueError(
                "Epsilon must be scalar (float) or batch-length tensor"
            )

    def _log_step(self,
                  x_orig: torch.Tensor,
                  labels: torch.Tensor,
                  delta: nn.Module,
                  grad: torch.Tensor,
                  class_loss: torch.Tensor,
                  aux_loss: torch.Tensor
                  ):
        """
        Using stored `writer`, log:
          - original inputs
          - adversarial inputs
          - over-the-air simulated adversarial inputs
          - loss values
          - parameter values
          - parameter gradients
        """

        tag = f'Filter-batch-{self._batch_id}'
        if self.writer is not None:

            x = x_orig.clone().detach()
            amps = delta.amplitudes.clone().detach()
            with torch.no_grad():
                x_adv = delta(x)
                outputs = self.pipeline(x_adv)

                success = self._compute_success_array(
                    x_clean=x,
                    x_adv=x_adv,
                    labels=labels
                )
                success_rate = torch.mean(1.0 * success)

            self.writer.log_loss(
                success_rate,
                f"{tag}/success-rate",
                global_step=self._i_max_iter
            )

            if self._i_max_iter == 0:  # original inputs
                self.writer.log_audio(
                    x[0],
                    f"{tag}/original",
                    global_step=0
                )

            self.writer.log_audio(  # adversarial inputs
                delta(x)[0],
                f"{tag}/adversarial",
                global_step=self._i_max_iter
            )

            with torch.no_grad():  # simulated original inputs
                simulated = self.pipeline.simulation(
                    x
                )[0]
            self.writer.log_audio(
                simulated,
                f"{tag}/simulated-original",
                global_step=self._i_max_iter
            )

            with torch.no_grad():  # simulated adversarial inputs
                simulated = self.pipeline.simulation(
                    delta(x)
                )[0]
            self.writer.log_audio(
                simulated,
                f"{tag}/simulated-adversarial",
                global_step=self._i_max_iter
            )

            # loss values
            self.writer.log_loss(
                class_loss,
                f"{tag}/classification-loss",
                global_step=self._i_max_iter
            )

            self.writer.log_loss(
                aux_loss,
                f"{tag}/auxiliary-loss",
                global_step=self._i_max_iter
            )

            # parameter values
            self.writer.log_filter(
                amps[0],
                f"{tag}/filter-amplitudes",
                global_step=self._i_max_iter
            )

            # parameter gradients
            grads = grad.reshape(amps.shape)
            self.writer.log_filter(
                grads[0],
                f"{tag}/filter-gradients",
                global_step=self._i_max_iter
            )

            # perturbation norm
            self.writer.writer.add_scalar(f"{tag}/perturbation-inf-norm",
                                          torch.max((x_adv - x)[0].abs()),
                                          global_step=self._i_max_iter)

            # perturbation norm
            self.writer.writer.add_scalar(f"{tag}/perturbation-2-norm",
                                          (x_adv - x)[0].norm(p=2),
                                          global_step=self._i_max_iter)

    def _attack_batch(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      eps: Union[float, torch.Tensor],
                      ):
        """
        Perform additive, selective, or orthogonal projected gradient descent
        attack on a batch of inputs.

        :param x: input tensor of shape (n_batch, ...)
        :param y: targets tensor of shape (n_batch, ...) in case of targeted
                  attack; original labels tensor of shape (n_batch, ...) in
                  case of untargeted attack
        """

        # require batch dimension
        assert x.ndim >= 2
        n_batch, signal_len = x.shape[0], x.shape[-1]
        x = x.reshape(n_batch, 1, signal_len)  # assume mono audio

        x_orig = x.clone().detach()
        x_final = x.clone().detach()

        # initialize filter perturbation controls
        delta = Filter(x_orig, block_size=self.block_size, n_bands=self.n_bands)

        _, n_frames, n_bands = delta.amplitudes.shape

        eps = eps.to(x) if eps is not None else eps

        loss_final = torch.zeros(n_batch).to(x) + float('inf')

        # if auxiliary loss is given, cache unperturbed inputs as reference to
        # avoid re-computing expensive intermediate representations
        if self.aux_loss is not None:
            self.aux_loss.set_reference(x_orig)

        if self.opt == 'adam':
            optimizer = torch.optim.Adam([delta.amplitudes],
                                         lr=self.lr,
                                         betas=(.99, .999),
                                         eps=1e-7,
                                         amsgrad=False)
        elif self.opt == 'lbfgs':
            optimizer = torch.optim.LBFGS([delta.amplitudes],
                                          lr=self.lr,
                                          line_search_fn='strong_wolfe')
        elif self.opt == 'sgd':
            optimizer = torch.optim.SGD([delta.amplitudes],
                                        lr=self.lr)
        else:
            raise ValueError(f'Invalid optimizer {self.opt}')

        # iteratively optimize perturbation
        for i in range(self.max_iter):

            self._i_max_iter = i

            # require gradients for perturbation parameters
            delta.amplitudes.requires_grad = True

            # randomly sample simulation parameters
            if self.eot_iter and not i % self.eot_iter:
                self.pipeline.sample_params()

            def closure():

                grad_total = torch.zeros(
                    n_batch, n_frames * n_bands
                ).to(x)

                outputs = self.pipeline(delta(x_orig))

                delta.amplitudes.grad = None

                class_scores = self.class_loss(outputs, y)
                class_loss = torch.mean(class_scores)
                class_loss.backward()
                class_loss_grad = delta.amplitudes.grad.detach()

                delta.amplitudes.grad = None

                if self.aux_loss is not None:
                    aux_scores = self.aux_loss(delta(x_orig), x_orig)
                    aux_loss = torch.mean(aux_scores)
                    aux_loss.backward()
                    aux_loss_grad = delta.amplitudes.grad.detach()
                else:
                    aux_scores = torch.zeros(n_batch).to(x)
                    aux_loss = torch.mean(aux_scores)
                    aux_loss_grad = torch.zeros_like(delta.amplitudes).detach()

                # toggle projections via success indicators
                shape = ((n_batch,) + (1,)*(len(grad_total.shape) - 1))

                # classifier evasion indicator, reshape for broadcasting
                class_success = class_scores <= 0.0
                class_success = 1.0 * class_success.view(shape)

                # save attacks which evade classifier and defense while
                # improving on best recorded auxiliary loss
                for j in range(n_batch):
                    if class_success[j].item() and \
                            loss_final[j].item() >= aux_scores[j].item():
                        x_final[j] = (delta(x_orig[j])).clone().detach()
                        loss_final[j] = aux_scores[j]

                # flatten gradient vectors along frame dimension to allow for
                # vector projection
                class_loss_grad = class_loss_grad.reshape(n_batch, -1)
                aux_loss_grad = aux_loss_grad.reshape(n_batch, -1)

                if self.mode is None or self.mode == 'none':  # use additive loss tradeoff
                    grad_total += class_loss_grad + aux_loss_grad
                elif self.mode == 'orthogonal':
                    if self.k and i % self.k:  # perform gradient projection every kth step
                        class_loss_grad_proj = class_loss_grad
                        aux_loss_grad_proj = aux_loss_grad
                    else:
                        class_loss_grad_proj = self._component_orthogonal(class_loss_grad,
                                                                          aux_loss_grad)

                        aux_loss_grad_proj = self._component_orthogonal(aux_loss_grad,
                                                                        class_loss_grad)

                    # only perform update 'along' a single loss per iteration
                    grad_total += class_loss_grad_proj * (1 - class_success)
                    grad_total += aux_loss_grad_proj * class_success

                elif self.mode == 'selective':
                    # only consider a single loss per iteration, without ensuring
                    # orthogonality to remaining loss terms
                    grad_total += class_loss_grad * (1 - class_success)
                    grad_total += aux_loss_grad * class_success
                else:
                    raise ValueError(f'Invalid attack mode {self.mode}')

                # regularize gradients
                if self.projection_norm in [2, float(2), "2"]:
                    grad_norms = torch.norm(grad_total.view(n_batch, -1), p=2, dim=-1) + 1e-20
                    grad_total = grad_total / grad_norms.view(n_batch, 1)
                elif self.projection_norm in [float("inf"), "inf"]:
                    grad_total = torch.sign(grad_total)
                elif self.projection_norm is None:
                    pass
                else:
                    raise ValueError(f'Invalid gradient regularization norm {self.projection_norm}')

                # set gradients
                delta.amplitudes.grad = grad_total.reshape(delta.amplitudes.shape)

                # log results
                self._log_step(x_orig,
                               y,
                               delta,
                               grad_total,
                               class_loss,
                               aux_loss
                               )

                return class_loss + aux_loss

            # optimizer step
            optimizer.step(closure)

            # project perturbation to feasible region
            if eps is None:
                pass
            elif self.projection_norm in [2, float(2), "2"]:
                norms = torch.norm(delta.amplitudes.clone().detach().view(n_batch, -1), p=2, dim=-1) + 1e-20
                factor = torch.min(
                    torch.tensor(1.0, dtype=norms.dtype, device=norms.device),
                    eps.reshape(eps.shape[0]) / norms
                ).view(-1, 1, 1)
                with torch.no_grad():
                    delta.amplitudes.mul_(factor)
            elif self.projection_norm in [float("inf"), "inf"]:
                with torch.no_grad():
                    delta.amplitudes.clamp_(min=-eps, max=eps)
            else:
                raise ValueError(f'Invalid projection norm {self.projection_norm}')

        return delta(x)
        #return torch.as_tensor(x_final)

    def attack(self,
               x: torch.Tensor,
               y: torch.Tensor
               ):

        # ensure projection bound `eps` is broadcastable
        self._check_compatibility_input_and_eps(x=x)

        x = x.to(self.pipeline.device)
        y = y.to(self.pipeline.device)

        adv_x = torch.clone(x)
        attack_success = torch.zeros(x.shape[0], dtype=torch.float)

        dataset = self._create_dataset(x, y)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        # hold attack success rates over multiple simulated environments
        success_rates = []

        # compute perturbation with batching
        for (batch_id, batch_all) in enumerate(data_loader):

            self._batch_id = batch_id

            (batch, batch_labels) = batch_all[0], batch_all[1]

            batch_index_1 = batch_id * self.batch_size
            batch_index_2 = (batch_id + 1) * self.batch_size

            # select batch projection bounds
            if isinstance(self.eps, torch.Tensor):
                if self.eps.shape[0] == x.shape[0]:
                    batch_eps = self.eps[batch_index_1:batch_index_2]
                else:
                    batch_eps = self.eps
            elif self.eps is None:
                batch_eps = self.eps
            else:
                batch_eps = torch.as_tensor(self.eps).reshape(1)

            adversarial_batch = self._attack_batch(
                x=batch,
                y=batch_labels,
                eps=batch_eps
            )

            if not self.eot_iter or not self.rand_evals:
                attack_success_batch = self._compute_success_array(
                    batch,
                    batch_labels,
                    adversarial_batch
                )
                success_rates.append(attack_success_batch.reshape(-1, 1).type(torch.float32))

            else:  # perform multiple random evaluations per input
                success_combined_batch = []

                pbar = tqdm(range(self.rand_evals))
                pbar.set_description(f'conducting random evaluations on batch {self._batch_id}')
                for i in pbar:
                    self.pipeline.sample_params()
                    rand_success_batch = self._compute_success_array(
                        batch,
                        batch_labels,
                        adversarial_batch
                    ).reshape(-1, 1)
                    success_combined_batch.append(rand_success_batch)

                attack_success_batch = (1.0 * torch.cat(
                    success_combined_batch, dim=-1
                )).mean(dim=-1)
                success_rates.append(attack_success_batch.reshape(-1, 1))

            adv_x[batch_index_1:batch_index_2] = adversarial_batch
            attack_success[batch_index_1:batch_index_2] = attack_success_batch

        return adv_x, attack_success
