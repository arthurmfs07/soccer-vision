import torch

# small patch to add coordinates channels


class AddCoords:
    """Given img: Tensor[C,H,W], return Tensor[C+2,H,W] with two extra channels:
    - channels C: x indices lineatly spaced 0-1
    - channels C+1: y indices linearly spaces 0-1
    """
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, H, W = img.shape
        device = img.device

        xs = torch.linspace(0, 1, W, device=device) \
            .view(1, 1, W).expand(1, H, W)
        ys = torch.linspace(0, 1, H, device=device) \
            .view(1, H, 1).expand(1, H, W)
        
        return torch.cat([img, xs, ys], dim=0)