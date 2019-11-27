import torch


def mask_image(image: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(image)
    mask[:, :, :, int(mask.size(-1) / 2) :] = 0
    return image * mask


def random_uniform(
    r1: int, r2: int, batch: int, dim: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(batch, dim).to(device)
