import torch

from ppocr.models.activation import Activation


class Head(torch.nn.Module):
    def __init__(self, in_channels):
        super(Head, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.conv_bn1 = torch.nn.BatchNorm2d(in_channels // 4)
        self.relu1 = Activation(act_type="relu")

        self.conv2 = torch.nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=2,
            stride=2,
        )
        self.conv_bn2 = torch.nn.BatchNorm2d(in_channels // 4)
        self.relu2 = Activation(act_type="relu")

        self.conv3 = torch.nn.ConvTranspose2d(
            in_channels=in_channels // 4, out_channels=1, kernel_size=2, stride=2
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        return x


class DBHead(torch.nn.Module):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    """

    def __init__(self, in_channels, k=50):
        super(DBHead, self).__init__()
        self.k = k

        self.binarize = Head(in_channels)
        self.thresh = Head(in_channels)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        shrink_maps = self.binarize(x)
        if not self.training:
            return shrink_maps

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat([shrink_maps, threshold_maps, binary_maps], dim=1)
        return y
