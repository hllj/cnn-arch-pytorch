import torch
from torchsummary import summary
from model.alexnet import AlexNet
from model.vgg import VGG, vgg_arch

if __name__ == "__main__":
    x = torch.randn(8, 3, 224, 224)
    model = VGG(100, vgg_arch["A"])
    # print(model(x).shape)
    summary(model, (3, 224, 224))
