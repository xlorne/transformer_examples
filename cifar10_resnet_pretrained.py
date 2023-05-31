import torch
import torch.nn as nn
import torchvision.models as models


def ResNet18():
    resnet18 = models.resnet18(pretrained=True)
    resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet18.maxpool = nn.Identity()
    resnet18.fc = nn.Linear(512, 10)
    return resnet18


if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)

    # 加载预训练的ResNet18模型
    resnet18 = ResNet18()
    print(resnet18)

    y = resnet18(x)
    print(y.size())
