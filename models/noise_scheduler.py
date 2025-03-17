import torch
import torch.nn as nn

class NoiseScheduler(nn.Module):
    def __init__(self, time_steps: int) -> None:
        super().__init__()
        self.betas = torch.linspace(0.0001,0.02,steps=time_steps)
        self.alphas = 1-self.betas
        self.alpha_ba = torch.cumprod(self.alphas, dim =0)#[-1]
        self.alpha_bar_sqrt = torch.sqrt(self.alpha_ba)
        self.one_minus_alpha_bar_sqrt = torch.sqrt(1 - self.alpha_ba)

    def forward(self, x: torch.Tensor, t: int):
        noise = torch.randn_like(x)
        alpha_bar_sqrt_t = self.alpha_bar_sqrt[t].view(-1, 1,1,1)
        one_minus_alpha_bar_sqrt_t = self.one_minus_alpha_bar_sqrt[t].view(-1,1,1,1)
        return alpha_bar_sqrt_t*x + one_minus_alpha_bar_sqrt_t*noise
