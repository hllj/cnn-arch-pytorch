import torch
from torch import nn

from base import BaseModel


vgg_arch = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "C": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "v1",
        "M",
        512,
        512,
        "v1",
        "M",
        512,
        512,
        "v1",
        "M",
    ],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(BaseModel):
    def __init__(self, num_of_classes, arch):
        super(VGG, self).__init__()
        self.num_of_classes = num_of_classes
        self.in_channels = 3
        self.arch = arch
        self.feature = self._create_from_arch(self.in_channels, arch)
        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.num_of_classes),
        )

    def forward(self, x):
        feat = self.feature(x)
        logits = self.classifer(feat)
        return logits

    def _create_from_arch(self, in_channels, arch):
        income_channels = in_channels
        conv_layers = []
        print("get here")
        for layer in arch:
            if isinstance(layer, int) is True:
                outcome_channels = layer
                conv_layers += [
                    nn.Conv2d(
                        income_channels,
                        outcome_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(),
                ]
                income_channels = outcome_channels
            elif layer == "M":
                conv_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif layer == "v1":
                conv_layers += [
                    nn.Conv2d(
                        income_channels,
                        income_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                    nn.ReLU(),
                ]
        return nn.Sequential(*conv_layers)
