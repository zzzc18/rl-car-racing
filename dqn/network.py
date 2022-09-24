import torch
import torch.nn as nn
import torchvision


class CarRacingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=7, padding=3, stride=2)  # 96x96x3->48x48x32
        self.pool1 = nn.MaxPool2d(
            kernel_size=2, stride=2)  # 48x48x32->24x24x32
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)  # 24x24x32->12x12x32
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1, stride=2)  # 12x12x32->6x6x64
        self.head1 = nn.Linear(2304, 512)
        self.head2 = nn.Linear(512, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.tanh(x)

        x = self.conv3(x)
        x = torch.tanh(x)

        x = x.view(x.shape[0], -1)
        x = self.head1(x)
        x = self.head2(x)
        return x


class CarRacingModelSmooth(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=7, padding=3, stride=2)  # 96x96x3->48x48x32
        self.pool1 = nn.MaxPool2d(
            kernel_size=2, stride=2)  # 48x48x32->24x24x32
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)  # 24x24x32->12x12x32
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1, stride=2)  # 12x12x32->6x6x64
        self.head1 = nn.Linear(2304, 512)
        self.head2 = nn.Linear(512, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.tanh(x)

        x = self.conv3(x)
        x = torch.tanh(x)

        x = x.view(x.shape[0], -1)
        x = self.head1(x)
        x = self.head2(x)
        return x


# class CarRacingModel(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.model = torchvision.models.resnet18(num_classes=5)

#     def forward(self, x):
#         x = self.model(x)
#         return x
