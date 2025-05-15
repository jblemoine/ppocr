import torch.nn as nn


class CTCHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(CTCHead, self).__init__()
        self.fc = nn.Linear(
            in_channels,
            out_channels,
            bias=True,
        )

    def forward(self, x):
        return self.fc(x)
