import torch
import torch.nn as nn

from src.simulation.simulation import Simulation
from src.models.speaker import SpeakerVerificationModel
from src.preprocess.preprocessor import Preprocessor

from typing import Union

################################################################################
# Differentiable end-to-end speaker verification pipeline
################################################################################


class Pipeline(nn.Module):

    def __init__(self,
                 model: SpeakerVerificationModel,
                 simulation: Simulation = None,
                 preprocessor: Preprocessor = None,
                 device: Union[str, torch.device] = 'cpu',
                 **kwargs
                 ):
        """
        Pipeline encompassing acoustic environment simulation, preprocessing,
        model, and defenses (purification and detection).

        :param model: the victim classifier
        :param simulation: an end-to-end differentiable acoustic simulation
        :param preprocessor: differentiable preprocessing stages
        :param device: store device to ensure all pipeline components are
                       correctly assigned
        """
        super().__init__()

        self.model = model
        self.simulation = simulation
        self.preprocessor = preprocessor
        self.device = device

        # flags to selectively enable pipeline stages
        self._enable_simulation = True
        self._enable_preprocessor = True

        # ensure model is in 'eval' mode
        self.model.eval()

        # move all submodules to stored device
        self.set_device(device)

        # freeze gradient computation for all stored parameters
        self._freeze_grad()

        # randomly initialize simulation parameters
        self.sample_params()

    @property
    def enable_simulation(self):
        return self._enable_simulation

    @enable_simulation.setter
    def enable_simulation(self, flag: bool):
        self._enable_simulation = flag

    @property
    def enable_preprocessor(self):
        return self._enable_preprocessor

    @enable_preprocessor.setter
    def enable_preprocessor(self, flag: bool):
        self._enable_preprocessor = flag

    def set_device(self, device: Union[str, torch.device]):
        """
        Move all submodules to stored device
        """
        self.device = device

        for module in self.modules():
            module.to(self.device)

    def _freeze_grad(self):
        """
        Disable gradient computations for all stored parameters
        """
        for p in self.parameters():
            p.requires_grad = False

    def sample_params(self):
        """
        Randomly re-sample the parameters of each stored effect
        """
        if self.simulation is not None:
            self.simulation.sample_params()

    def simulate(self, x: torch.Tensor):
        """
        Pass inputs through simulation
        """
        if self.enable_simulation and self.simulation is not None:
            x = self.simulation(x)

        return x

    def preprocess(self, x: torch.Tensor):
        """
        Pass inputs through preprocessing
        """
        if self.enable_preprocessor and self.preprocessor is not None:
            x = self.preprocessor(x)

        return x

    def forward(self, x: torch.Tensor):
        """
        Pass inputs through simulation, preprocessor, and model in sequence
        """
        x = self.simulate(x)
        x = self.preprocess(x)

        return self.model(x)

    def match_predict(self, y_pred: torch.tensor, y_true: torch.Tensor):
        """
        Determine whether target pairs are equivalent under stored model
        """
        return self.model.match_predict(y_pred, y_true)
