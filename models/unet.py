import torch
import torch.nn as nn
import torch.functional as F

class Unet(nn.Module):
    def __init__(self, in_channels: int=3, out_channels: int=3):
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


    def forward(self, x: torch.tensor):
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)

        x = self.bottleneck(x3)

        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        return self.out(x)

class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, pool: int=2, activation = nn.ReLU()):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
            # nn.MaxPool2d(pool),
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
        # print(f"x shape: {x.shape}, skip shape: {skip.shape}")
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

if __name__ =="__main__":
    x = torch.randn(30, 3, 256, 256)
    # cnn = DownSample(3,20, 5, 3,2)
    # up_cnn = UpSample(in_channels=20, out_channels=3, kernel_size=5, stride=3, pool=2)
    # down = cnn(x)
    # up = up_cnn(down)
    # print(out)
    # print(down.shape)
    # print(up.shape)
    # x = torch.randn(1, 3, 256, 256)
    model = Unet(in_channels=3, out_channels=3)
    # print(output)
    output = model(x)
    print(output.shape)
