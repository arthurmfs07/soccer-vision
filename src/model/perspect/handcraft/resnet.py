import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import *


@dataclass
class BlockSpec:
    out_channels: int
    num_blocks:   int
    stride:       int = 2 # stride of first block in stage


@dataclass
class ResNetConfig:
    input_channels:  int = 3
    stages: List[BlockSpec] = field(default_factory=lambda: [
        BlockSpec( 64, 2, stride=1),
        BlockSpec(128, 2, stride=2),
        BlockSpec(256, 2, stride=2),
        BlockSpec(512, 2, stride=2),
        ])
    use_stem:        int = True # toggle 7x7 stem
    activation_fn:   nn.Module = nn.ReLU(inplace=True)
    use_batchnorm:   bool = True
    fc_hidden:       int  = 1024
    out_dim:         int  = 8


class ResBlock(nn.Module):
    """2-conv residual block (similar to ResNet-18)"""
    expansion = 1

    def __init__(
            self, in_ch: int, out_ch: int, stride:int=1,
            activation: nn.Module = nn.ReLU(inplace=True),
            use_bn: bool = True
            ):
        super().__init__()

        bias = not use_bn
        BN   = (lambda c: nn.BatchNorm2d(c)) if use_bn else (lambda c: nn.Identity())

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=bias)
        self.bn1   = BN(out_ch)
        self.act   = activation
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=bias)
        self.bn2   = BN(out_ch)
    
        self.proj = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                BN(out_ch)
            )
        
    def forward(self, X):
        residual = self.proj(X)

        out = self.act(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))

        out += residual
        return self.act(out)


class ResNet(nn.Module):
    """
    input  : [B,3,H,W] (any H,W >= 32)
    output : [B,4,2] in [0,1]
    """

    def __init__(self, cfg: Optional[ResNetConfig] = None):
        super().__init__()
        self.cfg = cfg or ResNetConfig()

        act    = self.cfg.activation_fn
        use_bn = self.cfg.use_batchnorm
        in_ch = self.cfg.input_channels
        
        if self.cfg.use_stem:
            bias = not use_bn
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, 32, kernel_size=7, stride=2,
                          padding=3, bias=bias),
                nn.BatchNorm2d(32) if use_bn else nn.Identity(),
                act,
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            in_ch = 32
        else:
            self.stem = nn.Identity()

        stages = []        
        for spec in self.cfg.stages:
            blocks = [ResBlock(in_ch, spec.out_channels,
                               stride=spec.stride, activation=act,
                               use_bn=use_bn)]
            in_ch = spec.out_channels
            for _ in range(spec.num_blocks - 1):
                blocks.append(ResBlock(in_ch, in_ch,
                              stride=1, activation=act,
                              use_bn=use_bn))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, self.cfg.fc_hidden),
            act,
            nn.Linear(self.cfg.fc_hidden, self.cfg.out_dim)
        )

        self._initialize_weights()

    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Expect [B,3,H,W]"""
        X = self.stem(X)
        X = self.stages(X)
        X = self.avgpool(X)   # [B,C,1,1]
        X = self.fc(X)        # [B,8]
        X = torch.sigmoid(X)  # to [0..1]
        return X.view(-1, 4, 2)
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-1e3, b=1e-3)
                nn.init.constant_(m.bias, 0.0)


if __name__ == "__main__":
    model = ResNet()              # default config
    x = torch.randn(2, 3, 416, 416)
    y = model(x)
    print("output", y.shape)