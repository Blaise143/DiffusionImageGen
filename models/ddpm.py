import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, UNet2DModel
from .noise_scheduler import NoiseScheduler


class DenoisingModel(nn.Module):
    def __init__(self, time_steps: int, sample_size: int, in_channels: int=3, out_channels: int=3):
        super().__init__()
        self.noise_scheduler = NoiseScheduler(time_steps)
        self.unet =  UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=1,
            block_out_channels=(32, 64, 128),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )

    def forward(self, x: torch.Tensor, t: int):
        noisy_x = self.noise_scheduler(x, t)
        denoised_x = self.unet(noisy_x, timestep=t)
        return denoised_x


if __name__ == "__main__":
    a = torch.linspace(0.0001, 0.02, 1000)
    alpha = 1-a
    # b = torch.tensor([1,2,3, 4])
    import matplotlib.pyplot as plt
    plt.plot(a)
    plt.show()
    # print(torch.cumprod(alpha, dim=0))
    # print(torch.cumprod(b, dim=0))
