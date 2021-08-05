import torch
from torch import nn
import torch.nn.functional as F


# LeNet-5 Architecture: 1x32x32 -> 6 x (5x5), s=1, p=0 -> avg pool s=2, p=0
# -> 16 x (5x5) s=1, p=0 -> avg pool s=2, p=0 -> Flatten -> FC 120 -> FC 84 -> Output 10


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.p1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.c2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.p2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(400, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.sigmoid(self.c1(x))
        x = self.p1(x)

        x = torch.sigmoid(self.c2(x))
        x = self.p2(x)

        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)

        return output


if __name__ == "__main__":
    x = torch.randn(64, 1, 32, 32)
    model = LeNet()
    print(model(x).shape)
