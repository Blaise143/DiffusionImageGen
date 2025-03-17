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
        return self.alpha_bar_sqrt[t]*x + self.one_minus_alpha_bar_sqrt[t]*noise








if __name__ == "__main__":
    a = torch.linspace(0.0001, 0.02, 1000)
    alpha = 1-a
    # b = torch.tensor([1,2,3, 4])
    import matplotlib.pyplot as plt
    plt.plot(a)
    plt.show()
    # print(torch.cumprod(alpha, dim=0))
    # print(torch.cumprod(b, dim=0))
