import torch
from torch import nn
import torch.nn.functional as F

from base import BaseModel

# AlexNet Architecture:
# Input: 3 x 224 x 224 ->
# Layer 1: Conv 96 x (11, 11), stride=4; ReLU;
#           Local Response Norm; Max Pool 3x3 stride=2
# Layer 2: Conv 256 x (5, 5), stride=1, padding=2; ReLU;
#           Local Response Norm; Max Pool 3x3 stride=2
# Layer 3: Conv 384 x (3, 3), stride=1, padding=1; ReLU
# Layer 4: Conv 384 x (3, 3), stride=1, padding=1; ReLU
# Layer 5: Conv 256 x (3, 3), stride=1, padding=1; ReLU
#           Max Pool 3x3, stride=2
# Layer 6: FC 4096, ReLU
# Layer 7: FC 4096, ReLU
# Layer 8: num_of_classes


class AlexNet(BaseModel):
    def __init__(self, num_of_classes):
        super(AlexNet, self).__init__()
        self.num_of_classes = num_of_classes
        # Layer 1
        self.layer1_conv = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0
        )
        self.layer1_lrn = nn.LocalResponseNorm(size=5, k=2.0, alpha=1e-4, beta=0.75)
        self.layer1_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Layer 2
        self.layer2_conv = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
        )
        self.layer2_lrn = nn.LocalResponseNorm(size=5, k=2.0, alpha=1e-4, beta=0.75)
        self.layer2_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Layer 3
        self.layer3_conv = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
        )

        # Layer 4
        self.layer4_conv = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
        )

        # Layer 5
        self.layer5_conv = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.layer5_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.flatten = nn.Flatten()

        # Layer 6
        self.fc6 = nn.Linear(6 * 6 * 256, 4096)

        # Layer 7
        self.fc7 = nn.Linear(4096, 4096)

        # Layer 8
        self.fc8 = nn.Linear(4096, self.num_of_classes)

    def forward(self, x):
        # Layer 1
        x = F.relu(self.layer1_conv(x))
        x = self.layer1_lrn(x)
        x = self.layer1_maxpool(x)
        print("layer 1:", x.shape)

        # Layer 2
        x = F.relu(self.layer2_conv(x))
        x = self.layer2_lrn(x)
        x = self.layer2_maxpool(x)
        print("layer 2:", x.shape)

        # Layer 3
        x = F.relu(self.layer3_conv(x))
        print("layer 3:", x.shape)

        # Layer 4
        x = F.relu(self.layer4_conv(x))
        print("layer 4:", x.shape)

        # Layer 5
        x = F.relu(self.layer5_conv(x))
        x = self.layer5_maxpool(x)
        print("layer 5:", x.shape)

        x = self.flatten(x)

        # Layer 6
        x = self.fc6(x)
        print("layer 6:", x.shape)

        # Layer 7
        x = self.fc7(x)
        print("layer 7:", x.shape)

        # Layer 8
        x = self.fc8(x)
        print("layer 8:", x.shape)

        output = F.log_softmax(x, dim=1)
        return output
