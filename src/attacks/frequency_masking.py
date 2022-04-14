import torch
import torch.nn as nn

from tqdm import tqdm

from src.pipelines import Pipeline
from src.writer import Writer
from src.loss.adversarial import AdversarialLoss
from src.loss.frequency_masking import FrequencyMaskingLoss

from typing import Optional, Union, TYPE_CHECKING

################################################################################
# Frequency-masking attack proposed by Qin et al.
################################################################################


class FrequencyMaskingAttack:

    def __init__(self,
                 pipeline: Pipeline,
                 class_loss: AdversarialLoss = None,
                 max_iter_1: int = 1000,
                 max_iter_2: int = 4000,
                 opt_1: str = 'sgd',
                 opt_2: str = 'sgd',
                 lr_1: float = 0.003,
                 lr_2: float = 1e-5,
                 alpha: float = 0.05,
                 loss_theta_min: float = 0.05,
                 decrease_factor_eps: float = 0.8,
                 n_iter_decrease_eps: int = 10,
                 increase_factor_alpha: float = 1.2,
                 n_iter_increase_alpha: int = 20,
                 decrease_factor_alpha: float = 0.8,
                 n_iter_decrease_alpha: int = 50,
                 eps: Union[int, float, torch.Tensor] = 0.06,
                 eot_iter: int = 0,
                 batch_size: int = 32,
                 rand_evals: int = 0,
                 writer: Writer = None
                 ):
        """
        Adapted from `Imperceptible, Robust, and Targeted Adversarial Examples
        for Automatic Speech Recognition` by Qin et al.
        (https://arxiv.org/abs/1903.10346) and the Adversarial Robustness
        Toolkit (ART) implementation of the `imperceptible_asr` attack
        (https://github.com/Trusted-AI/adversarial-robustness-toolbox)

        :param pipeline: a Pipeline object wrapping a defended classifier
        :param class_loss: a adversarial loss object; must take model
                           predictions and targets as arguments
        :param writer: a Writer for logging attack progress
        """
        self.pipeline = pipeline
        self.class_loss = class_loss

        self.eps = eps
        self.max_iter_1 = max_iter_1
        self.max_iter_2 = max_iter_2
        self.lr_1 = lr_1
        self.lr_2 = lr_2

        self.alpha = alpha
        self.loss_theta_min = loss_theta_min

        self.decrease_factor_eps = decrease_factor_eps
        self.n_iter_decrease_eps = n_iter_decrease_eps
        self.increase_factor_alpha = increase_factor_alpha
        self.n_iter_increase_alpha = n_iter_increase_alpha
        self.decrease_factor_alpha = decrease_factor_alpha
        self.n_iter_decrease_alpha = n_iter_decrease_alpha

        self.opt_1 = opt_1
        self.opt_2 = opt_2

        self.batch_size = batch_size

        self.eot_iter = eot_iter
        self.rand_evals = rand_evals

        self.writer = writer

        self._batch_id = 0
        self._i_max_iter = 0

        self._check_loss(class_loss)

        # initialize masking loss
        self.loss_theta = FrequencyMaskingLoss(
            reduction=None,
            alpha=1.0,
            pad=True,
            normalize='peak'
        )

    @staticmethod
    def _check_loss(class_loss: AdversarialLoss):
        try:
            assert class_loss is not None
        except AssertionError:
            raise ValueError('Must provide classification loss')

        try:
            assert class_loss.reduction is None or class_loss.reduction == 'none'
        except AssertionError:
            raise ValueError('All losses must provide unreduced scores')

    @staticmethod
    def _create_dataset(x: torch.Tensor, y: torch.Tensor):

        dataset = torch.utils.data.TensorDataset(
            x.type(torch.float32),
            y.type(torch.float32)
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

    def _check_compatibility_input_and_eps(self, x: torch.Tensor):
        """
        Check the compatibility of the input with `eps` of the same shape.
        :param x: An array with the original inputs.
        """
        if isinstance(self.eps, torch.Tensor):
            # Ensure the eps array is broadcastable
            if self.eps.ndim > x.ndim:
                raise ValueError("The `eps` shape must be broadcastable to input shape.")
            elif self.eps.shape[0] != x.shape[0] and self.eps.shape[0] != 1:
                raise ValueError("The `eps` shape must be broadcastable to input shape.")
            elif self.eps.ndim < x.ndim:
                for i in range(x.ndim - self.eps.ndim):
                    self.eps = self.eps.unsqueeze(-1)  # unsqueeze to allow broadcasting

    def _log_step_1(self,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    delta: torch.Tensor,
                    loss_1_grad: torch.Tensor,
                    class_loss: Union[float, torch.Tensor],
                    ):
        """
        Using stored `writer`, log:
          - original inputs
          - adversarial inputs
          - over-the-air simulated adversarial inputs
          - loss values
          - predictions
          - parameter gradients
        """
        tag = f'Qin-stage-1-batch-{self._batch_id}'

        if self.writer is not None:

            x = x.clone().detach()
            delta = delta.clone().detach()
            with torch.no_grad():
                outputs = self.pipeline(x + delta)

                success = self._compute_success_array(
                    x_clean=x,
                    x_adv=x + delta,
                    labels=y
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
                (delta+x)[0],
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
                    (delta+x)
                )[0]
            self.writer.log_audio(
                simulated,
                f"{tag}/simulated-adversarial",
                global_step=self._i_max_iter
            )

            # perturbation norm
            self.writer.writer.add_scalar(f"{tag}/perturbation-inf-norm",
                                          torch.max(delta[0].abs()),
                                          global_step=self._i_max_iter)

            # perturbation norm
            self.writer.writer.add_scalar(f"{tag}/perturbation-2-norm",
                                          delta[0].norm(p=2),
                                          global_step=self._i_max_iter)

            # gradients
            self.writer.log_gradient(loss_1_grad[0],
                                     f"{tag}/gradients-combined",
                                     global_step=self._i_max_iter)

            # loss value
            self.writer.log_loss(
                class_loss,
                f"{tag}/classification-loss",
                global_step=self._i_max_iter
            )

    def _log_step_2(self,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    delta: torch.Tensor,
                    loss_2_grad: torch.Tensor,
                    class_loss: Union[float, torch.Tensor],
                    theta_loss: Union[float, torch.Tensor]
                    ):
        """
        Using stored `writer`, log:
          - original inputs
          - adversarial inputs
          - over-the-air simulated adversarial inputs
          - loss values
          - predictions
          - parameter gradients
        """
        tag = f'Qin-stage-2-batch-{self._batch_id}'
        if self.writer is not None:

            x = x.clone().detach()
            delta = delta.clone().detach()
            with torch.no_grad():
                outputs = self.pipeline(x + delta)

                success = self._compute_success_array(
                    x_clean=x,
                    x_adv=x + delta,
                    labels=y
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
                    global_step=0 + self.max_iter_1
                )

            self.writer.log_audio(  # adversarial inputs
                (delta+x)[0],
                f"{tag}/adversarial",
                global_step=self._i_max_iter + self.max_iter_1
            )

            with torch.no_grad():  # simulated original inputs
                simulated = self.pipeline.simulation(
                    x
                )[0]
            self.writer.log_audio(
                simulated,
                f"{tag}/simulated-original",
                global_step=self._i_max_iter + self.max_iter_1
            )

            with torch.no_grad():  # simulated adversarial inputs
                simulated = self.pipeline.simulation(
                    (delta+x)
                )[0]
            self.writer.log_audio(
                simulated,
                f"{tag}/simulated-adversarial",
                global_step=self._i_max_iter + self.max_iter_1
            )

            # perturbation norm
            self.writer.writer.add_scalar(f"{tag}/perturbation-inf-norm",
                                          torch.max(delta[0].abs()),
                                          global_step=self._i_max_iter + self.max_iter_1)

            # perturbation norm
            self.writer.writer.add_scalar(f"{tag}/perturbation-2-norm",
                                          delta[0].norm(p=2),
                                          global_step=self._i_max_iter + self.max_iter_1)

            # gradients
            self.writer.log_gradient(loss_2_grad[0],
                                     f"{tag}/gradients-combined",
                                     global_step=self._i_max_iter + self.max_iter_1)

            # loss values
            self.writer.log_loss(
                class_loss,
                f"{tag}/classification-loss",
                global_step=self._i_max_iter + self.max_iter_1
            )
            self.writer.log_loss(
                theta_loss,
                f"{tag}/perceptual-loss",
                global_step=self._i_max_iter + self.max_iter_1
            )

    def _attack_batch(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      eps: Union[int, float, torch.Tensor]):
        """
        Perform imperceptible attack on a batch of inputs.

        :param x: input tensor of shape (n_batch, ...)
        :param y: targets tensor of shape (n_batch, ...) in case of targeted
                  attack; original labels tensor of shape (n_batch, ...) in
                  case of untargeted attack
        """

        delta = torch.zeros_like(x)
        x_orig = x.clone().detach()

        x_adversarial = x.clone().detach()  # store output of stage 1
        x_imperceptible = x.clone().detach()  # store output of stage 2

        n_batch = x.shape[0]

        # tile epsilon to batch dimension
        eps = eps.to(x)
        if eps.shape[0] == 1:
            eps = torch.repeat_interleave(
                eps, repeats=n_batch, dim=0
            ).reshape(-1, 1, 1)

        # cache masking thresholds for unperturbed inputs as reference to avoid
        # re-computing expensive intermediate representations
        self.loss_theta.set_reference(x_orig)

        # stage 1: compute adversarial input without perceptual regularization
        if self.opt_1 == 'adam':
            optimizer = torch.optim.Adam([delta],
                                         lr=self.lr_1,
                                         betas=(.99, .999),
                                         eps=1e-7,
                                         amsgrad=False)
        elif self.opt_1 == 'lbfgs':
            optimizer = torch.optim.LBFGS([delta],
                                          lr=self.lr_1,
                                          line_search_fn='strong_wolfe')
        elif self.opt_1 == 'sgd':
            optimizer = torch.optim.SGD([delta],
                                        lr=self.lr_1)
        else:
            raise ValueError(f'Invalid stage-1 optimizer {self.opt_1}')

        # perform stage-1 optimization
        for i in range(self.max_iter_1):

            self._i_max_iter = i

            delta.requires_grad = True

            if self.eot_iter and not i % self.eot_iter:  # randomly sample simulation parameters
                self.pipeline.sample_params()

            def closure():

                outputs = self.pipeline(x_orig + delta)

                delta.grad = None

                class_scores = self.class_loss(outputs, y)
                class_loss = torch.mean(class_scores)
                class_loss.backward(retain_graph=True)
                class_loss_grad = delta.grad.detach()

                # sum loss gradients and take sign
                loss_1_grad = torch.sign(class_loss_grad)

                # classifier evasion indicator, reshape for broadcasting
                class_success = class_scores <= 0.0
                class_success = 1.0 * class_success

                # save successful attacks and update epsilon bound
                if not i % self.n_iter_decrease_eps:

                    for j in range(n_batch):
                        if class_success[j].item():
                            perturbation_norm = torch.max(torch.abs(delta[j]))
                            if eps[j] > perturbation_norm:
                                eps[j] = perturbation_norm
                            eps[j] *= self.decrease_factor_eps
                            x_adversarial[j] = (x_orig[j] + delta[j]).clone().detach()

                # apply gradients
                delta.grad = loss_1_grad

                # log results
                if self.writer is not None:
                    self._log_step_1(
                        x_orig,
                        y,
                        delta,
                        loss_1_grad,
                        class_loss
                    )

                return class_loss

            # optimizer step
            optimizer.step(closure)

            # clip perturbation to feasible region
            with torch.no_grad():
                delta.clamp_(min=-eps, max=eps)

        # stage 2: optimize adversarial and perceptual objectives simultaneously
        delta = delta.detach()
        x_adversarial = (x_orig + delta)

        # stage 1: compute adversarial input without perceptual regularization
        if self.opt_2 == 'adam':
            optimizer = torch.optim.Adam([delta],
                                         lr=self.lr_2,
                                         betas=(.99, .999),
                                         eps=1e-7,
                                         amsgrad=False)
        elif self.opt_2 == 'lbfgs':
            optimizer = torch.optim.LBFGS([delta],
                                          lr=self.lr_2,
                                          line_search_fn='strong_wolfe')
        elif self.opt_2 == 'sgd':
            optimizer = torch.optim.SGD([delta],
                                        lr=self.lr_2)
        else:
            raise ValueError(f'Invalid stage-1 optimizer {self.opt_2}')

        # set lower bound on loss tradeoff factor
        alpha_min = 0.0005

        # tile alpha to batch dimension and reshape to allow broadcasting with gradients
        alpha = torch.tensor(self.alpha).to(x).reshape(1)
        alpha = torch.repeat_interleave(alpha, repeats=n_batch, dim=0)
        shape = ((n_batch,) + (1,)*(len(delta.shape) - 1))
        alpha = alpha.reshape(shape)

        # store running perceptual loss scores
        theta_scores_best = (torch.ones(n_batch) * float('inf')).to(x)

        # perform stage-2 optimization
        for i in range(self.max_iter_2):

            self._i_max_iter = i

            delta.requires_grad = True

            if self.eot_iter and not i % self.eot_iter:  # randomly sample simulation parameters
                self.pipeline.sample_params()

            def closure():

                outputs = self.pipeline(x_orig + delta)

                delta.grad = None

                class_scores = self.class_loss(outputs, y)
                class_loss = torch.mean(class_scores)
                class_loss.backward(retain_graph=True)
                class_loss_grad = delta.grad.detach()

                delta.grad = None

                theta_scores = self.loss_theta(x_orig + delta, x_orig)
                theta_loss = torch.mean(theta_scores)
                theta_loss.backward()
                theta_loss_grad = delta.grad.detach()

                loss_threshold_mask = ((theta_scores > self.loss_theta_min) * 1.0).reshape(shape)

                # do not apply perceptual loss once threshold is met
                theta_loss_grad *= loss_threshold_mask

                # weighted sum of loss gradients
                loss_2_grad = class_loss_grad + alpha * theta_loss_grad

                # classifier evasion indicator, reshape for broadcasting
                class_success = class_scores <= 0.0
                class_success = 1.0 * class_success

                # save successful attacks and update loss tradeoff parameter
                if not i % self.n_iter_increase_alpha or not i % self.n_iter_decrease_alpha:

                    for j in range(n_batch):

                        # if successful, increase perceptual weight
                        if class_success[j].item() and \
                                not i % self.n_iter_increase_alpha:
                            alpha[j] *= self.increase_factor_alpha  # increase alpha
                            # save current best attack
                            if theta_scores[j] < theta_scores_best[j]:
                                x_imperceptible[j] = (x_orig[j] + delta[j]).clone().detach()
                                theta_scores_best[j] = theta_scores[j]

                        # if unsuccessful, decrease perceptual weight
                        elif not i % self.n_iter_decrease_alpha:
                            alpha[j] *= self.decrease_factor_alpha
                            alpha[j].clamp_(min=alpha_min)

                # apply gradients
                delta.grad = loss_2_grad

                # log results
                if self.writer is not None:
                    self._log_step_2(
                        x_orig,
                        y,
                        delta,
                        loss_2_grad,
                        class_loss,
                        theta_loss
                    )

                return class_loss + theta_loss

            optimizer.step(closure)

        return x_orig + delta

    def attack(self, x: torch.Tensor,
               y: torch.Tensor,
               **kwargs):

        # Ensure eps is broadcastable
        self._check_compatibility_input_and_eps(x=x)

        x = x.to(self.pipeline.device)
        y = y.to(self.pipeline.device)

        adv_x = torch.clone(x)
        attack_success = torch.zeros(x.shape[0], dtype=torch.float)

        dataset = self._create_dataset(x, y)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # compute success rates, optionally over multiple simulated environments
        success_rates = []

        # Compute perturbation with batching
        for (batch_id, batch_all) in enumerate(data_loader):

            self._batch_id = batch_id
            (batch, batch_labels) = batch_all[0], batch_all[1]

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size

            # Compute batch_eps
            if isinstance(self.eps, torch.Tensor):
                if self.eps.shape[0] == x.shape[0]:
                    batch_eps = self.eps[batch_index_1:batch_index_2]
                else:
                    batch_eps = self.eps
            else:
                batch_eps = torch.as_tensor(self.eps).reshape(1)

            adversarial_batch = self._attack_batch(
                x=batch, y=batch_labels, eps=batch_eps
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

                # for now, save attacks that succeed more than 50% of the time
                attack_success_batch = (1.0 * torch.cat(
                    success_combined_batch, dim=-1
                )).mean(dim=-1)
                success_rates.append(attack_success_batch.reshape(-1, 1))

            adv_x[batch_index_1:batch_index_2] = adversarial_batch
            attack_success[batch_index_1:batch_index_2] = attack_success_batch

        return adv_x, attack_success

