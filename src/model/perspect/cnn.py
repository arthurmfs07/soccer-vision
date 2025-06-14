import torch
from torch import nn
from dataclasses import dataclass, field
from typing import List

@dataclass
class CNNConfig:
    """
    - input_channels: e.g. 3 for RGB
    - hidden_channels: last layer is n of output feature_maps (size n)
    - kernel_sizes: (size n)
    - strides:      (size n)
    - paddings:     (size n)
    - use_batchnorm: if true, add BatchNorm2d
    - activation_fn: activation function after each conv
    """
    input_channels: int = 3
    hidden_channels: List[int] = field(default_factory=lambda:  [ 32,  64, 128, 256])
    # hidden_channels: List[int] = field(default_factory=lambda:  [ 64, 128, 256, 512])
    kernel_sizes:    List[int] = field(default_factory= lambda: [  3,   3,   3,   3])
    strides:         List[int] = field(default_factory=lambda:  [  2,   2,   2,   1])
    paddings:        List[int] = field(default_factory=lambda:  [  1,   1,   1,   1])
    activation_fn: nn.Module   = nn.LeakyReLU(negative_slope=0.01)
    use_batchnorm: bool        = True


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.config = CNNConfig()

        hidden_channels = self.config.hidden_channels
        kernel_sizes    = self.config.kernel_sizes
        strides         = self.config.strides
        paddings        = self.config.paddings
        input_channels  = self.config.input_channels
        use_batchnorm   = self.config.use_batchnorm
        activation_fn   = self.config.activation_fn

        assert len(self.config.hidden_channels) > 0, "hidden_channels must be non-empty"        
        assert len(self.config.hidden_channels) == len(self.config.kernel_sizes), "kernel_sizes must match hidden_channels in size"
        assert len(self.config.hidden_channels) == len(self.config.strides),      "strides must match hidden_channels in size"
        assert len(self.config.hidden_channels) == len(self.config.paddings),     "paddings must match hidden_channels in size"

        layers = []
        in_ch = input_channels
        for (out_ch,k,s,p) in zip(
            hidden_channels, 
            kernel_sizes, 
            strides, 
            paddings
        ):
            layers.append(nn.Conv2d(in_ch, out_ch, k, s, p))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(activation_fn)
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        pool_shape = (4, 4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(pool_shape)

        self.feature_dim = in_ch * pool_shape[0] * pool_shape[1]

        self._initialize_weights()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: [B, feature_dim]
        """
        X = self.conv_layers(X)
        X = self.adaptive_pool(X)
        return X.flatten(1) # [B, feature_dim]


    


    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



if __name__ == "__main__":

    model = CNN()
    input_tensor = torch.rand(1, 3, 416, 416) # B x C x W x H

    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)