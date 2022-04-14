import torch
import torch.nn as nn
import torchaudio

from src.models.resnetse34v2.se_blocks import *
from src.constants import *

from typing import Iterable

################################################################################
# ResNetSE34V2 architecture for speaker embedding
################################################################################


class ResNetSE34V2(nn.Module):
    """
    ResNetSE34V2 model proposed in Heo et al. (arXiv: 2009.14153). Code adapted
    from https://github.com/clovaai/voxceleb_trainer.
    """
    def __init__(self,
                 layers: Iterable[int] = (3, 4, 6, 3),
                 num_filters: Iterable[int] = (32, 64, 128, 256),
                 nOut: int = 512,
                 encoder_type: str = 'ASP',
                 n_mels: int = 64,
                 log_input: bool = True):
        """
        Squeeze-and-Excitation ResNet architecture for speaker embedding.
        Accepts 16kHz waveform audio input of arbitrary length, converts to
        spectrogram, and applies four residual/SE convolutional blocks followed
        by a linear layer to produce discriminative embeddings.

        :param layers: an iterable containing the kernel dimension of each SE
                       layer; must have length 4 (one entry per SE layer)
        :param num_filters: an iterable containing the number of filters /
                            output channels of each SE layer; must have length 4
                            (one entry per SE layer)
        :param nOut: final embedding dimension
        :param encoder_type: method of aggregating frame-level features into
                             utterance-level features. Must be one of "SAP"
                             (self-attentive pooling) or "ASP" (attentive
                             statistics pooling)
        :param n_mels: mel bins for spectrogram
        :param log_input: if True, apply log to spectrogram
        """
        super().__init__()

        # enforce block type
        block = SEBasicBlock

        assert len(layers) == 4
        assert len(num_filters) == 4

        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels = n_mels
        self.log_input = log_input

        # prior to SE layers, input spectrogram is passed through a "vanilla"
        # convolutional layer
        self.conv1 = nn.Conv2d(
            1,
            num_filters[0],
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(
            block,
            num_filters[1],
            layers[1],
            stride=(2, 2)
        )
        self.layer3 = self._make_layer(
            block,
            num_filters[2],
            layers[2],
            stride=(2, 2)
        )
        self.layer4 = self._make_layer(
            block,
            num_filters[3],
            layers[3],
            stride=(2, 2)
        )

        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.torchfb = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=SR,
                n_fft=512,
                win_length=400,
                hop_length=160,
                window_fn=torch.hamming_window,
                n_mels=n_mels)
        )

        outmap_size = int(self.n_mels/8)

        # attention block: collapse and restore channel dimension of feature
        # maps through 1x1 convolutions, then pass frame/time dimension through
        # softmax to obtain a frame-wise weighting of each channel. For more
        # details, see Zhu et al. (https://bit.ly/3E10jBT)
        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        # for self-attentive pooling
        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        # for attentive statistics pooling
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError(f'Undefined encoder {self.encoder_type}')

        self.fc = nn.Linear(out_dim, nOut)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x: torch.Tensor):

        x = self.torchfb(x)+1e-6
        if self.log_input:
            x = x.log()
        x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # consolidate channel and frequency dimensions, leaving time dimension
        x = x.reshape(x.size()[0], -1, x.size()[-1])  # (n_batch, -1, time)

        w = self.attention(x)

        if self.encoder_type == "SAP":

            # apply attention weights and take weighted average over time
            x = torch.sum(x * w, dim=2)  # (n_batch, outmap_size)

        elif self.encoder_type == "ASP":

            # apply attention weights and take weighted average over time
            mu = torch.sum(x * w, dim=2)  # (n_batch, outmap_size)

            # compute standard deviation from weighted means
            sg = torch.sqrt(
                (
                        torch.sum((x**2) * w, dim=2) - mu**2
                ).clamp(min=1e-5)
            )  # (n_batch, outmap_size)
            x = torch.cat((mu, sg), 1)  # (n_batch, 2 * outmap_size)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)  # (n_batch, nOut)

        return x
