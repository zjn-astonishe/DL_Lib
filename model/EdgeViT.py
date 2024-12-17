import torch
import torch.nn as nn
from torch.nn import ConvTranspose2d

class LocalAgg(nn.Module):
    """用于聚合局部特征"""
    def __init__(
        self, 
        channels: int
    ) -> None:
        
        super().__init__()

        block = nn.Sequential()
        block.add_module(
            name='pointwise_prenorm_0',
            module=nn.BatchNorm2d(channels)
        )
        block.add_module(
            name='pointwise_conv_0',
            module=nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                bias=False
            )
        )
        block.add_module(
            name='depthwise_conv',
            module=nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                padding=1,
                kernel_size=3,
                groups=channels,
                bias=False
            )
        )
        block.add_module(
            name='pointwise_prenorm_1',
            module=nn.BatchNorm2d(channels)
        )
        block.add_module(
            name='pointwise_conv_1',
            module=nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                bias=False
            )
        )

        self.block = block
        self.channels = channels

    def forward(self, X):
        """
        [B, C, H, W] = X.shape
        """
        X = self.block(X)
        return X

class LocalAgg1(nn.Module):
    """用于聚合局部特征"""
    def __init__(
        self, 
        channels: int
    ) -> None:
        
        super().__init__()

        block = nn.Sequential()
        block.add_module(
            name='pointwise_prenorm_0',
            module=nn.BatchNorm2d(channels)
        )
        block.add_module(
            name='pointwise_conv_0',
            module=nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                bias=False
            )
        )
        block.add_module(
            name='depthwise_conv',
            module=nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                groups=channels,
                bias=False
            )
        )
        block.add_module(
            name='pointwise_prenorm_1',
            module=nn.BatchNorm2d(channels)
        )
        block.add_module(
            name='pointwise_conv_1',
            module=nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                bias=False
            )
        )

        self.block = block
        self.channels = channels

    def forward(self, X):
        """
        [B, C, H, W] = X.shape
        """
        X = self.block(X)
        return X

class LocalProp(nn.Module):
    """用于分散局部特征"""
    def __init__(
        self, 
        channels,
        sample_rate
    ) -> None:
        super().__init__()
        self.localprop = ConvTranspose2d(channels, channels, kernel_size=sample_rate, stride=sample_rate, groups=channels)
    
    def forward(self, X):
        X = self.localprop(X)
        return X

if __name__== "__main__" :
    X = torch.rand(2, 1, 3, 3)
    X1 = torch.rand(968, 16, 3, 3)
    model = LocalAgg(1)
    model1 = LocalAgg1(16)
    model2 = LocalProp(16, sample_rate=3)
    print(X1)
    # print(model(X))
    X1 = model1(X1)
    print(X1.shape)
    X2 = model2(X1)
    print(X2.shape)