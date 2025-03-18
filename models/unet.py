import torch
import torch.nn as nn
import math

class Unet(nn.Module):
    def __init__(self, in_channels: int=3, out_channels: int=3, time_embeddig_dim: int=64):
        super().__init__()
        self.down1 = DownSample(in_channels, 32)
        self.down2 = DownSample(32, 64)
        self.down3 = DownSample(64, 128)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 264, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(264, 264, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = UpSample(264,128, skip_channels=128)
        self.up2 = UpSample(128, 64, skip_channels=64)
        self.up3 = UpSample(64, 32, skip_channels=32)

        self.out = nn.Conv2d(in_channels=32,out_channels=out_channels, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(time_embeddig_dim, 264),
            nn.ReLU(),
            nn.Linear(264, 264),
            nn.ReLU()
        )
        self.time_embedding_dim = time_embeddig_dim


    def forward(self, x: torch.Tensor, timestep: int):

        time_embedding = sinusoidal_embedding(t = timestep, dim= self.time_embedding_dim).unsqueeze(0).repeat(x.shape[0],1)

        time_embedding = self.mlp(time_embedding).unsqueeze(-1).unsqueeze(-1)

        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        print(f"time_embedding shape: {time_embedding.shape}")
        x = self.bottleneck(x3)+time_embedding
        print(f"bottleneck dim: {x.shape}")

        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        return self.out(x)

class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int=3, stride: int=1, pool: int=2,
                 activation = nn.ReLU()):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=kernel_size // 2),
        )
        self.pool = nn.MaxPool2d(pool)
    def forward(self, x: torch.Tensor):
        skip = self.convs(x)
        x = self.pool(skip)
        return x, skip

class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, kernel_size: int=3,stride=2, pool: int=2, activation=nn.ReLU()):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1, output_padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            activation
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def sinusoidal_embedding(t: int, dim: int):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t])
    else:
        t = t.float()
    t = t.unsqueeze(1) if t.dim() == 1 else t
    assert dim % 2 == 0, "Embedding dim not even"
    half_dim = dim // 2
    exp = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / (half_dim - 1)
    emb = t * torch.exp(exp)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if emb.shape[0] == 1:
        emb = emb.squeeze(0)
    return emb


if __name__ =="__main__":

    x = torch.randn(30, 3, 256, 256)
    # y = torch.tensor(3.)
    # print(x.float())
    m = sinusoidal_embedding(3, 40)
    # print(m.shape)
    model = Unet(in_channels=3, out_channels=3)
    output = model(x, 2)
    print(output.shape)
    # a = torch.tensor(3.).unetsqueeze(-1)
    # print(a)
    # print(a.unsqueeze(-1))
    # print(torch.zeros((3,)).unsqueeze(0).repeat(a.shape[0],1))
