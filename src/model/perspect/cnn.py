import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass, field

from typing import List, Optional

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
    hidden_channels: List[int] = field(default_factory=lambda: [ 32,  64,  96])
    kernel_sizes:    List[int] = field(default_factory=lambda: [  4,   4,   4])
    strides:         List[int] = field(default_factory=lambda: [  2,   2,   2])
    paddings:        List[int] = field(default_factory=lambda: [  1,   1,   1])
    use_batchnorm: bool        = True
    activation_fn: nn.Module   = nn.ReLU()
    image_size: tuple          = (416, 416)


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
        in_channels = input_channels
        for (out_channels, kernel_size, stride, padding) in zip(hidden_channels, kernel_sizes, strides, paddings):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_fn)
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        dummy_input = torch.zeros(1, input_channels, *self.config.image_size)
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)
        self._flattened_size = dummy_output.view(1, -1).shape[1]

        self.fc = nn.Linear(self._flattened_size, 8)
        self._initialize_weights()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv_layers(X)
        X = self.flatten(X)
        X = self.fc(X)  # shape [B, 8] (-1, 1)
        return X
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(self.fc.weight, a=-1e-3, b=1e-3)
                nn.init.constant_(self.fc.bias, 0.0)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)



if __name__ == "__main__":

    model = CNN()
    input_tensor = torch.rand(1, 3, 416, 416) # B x C x W x H

    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)