import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class QuantizationLayer(nn.Module):
    """
    미분 가능한 양자화 시뮬레이션 (Straight-Through Estimator)
    """
    def forward(self, x):
        if self.training:
            return x + (x.round() - x).detach()
        else:
            return x.round()

class Compressor(nn.Module):
    """Skip Connection 압축 모듈 (1x1 Conv + Quantization)"""
    def __init__(self, in_channels, compressed_channels):
        super().__init__()
        self.channel_reducer = nn.Conv2d(in_channels, compressed_channels, kernel_size=1)
        self.quantizer = QuantizationLayer()
        self.bn = nn.BatchNorm2d(compressed_channels)

    def forward(self, x):
        x = self.channel_reducer(x)
        x = self.bn(x)
        # 실제로는 여기서 int8 범위 스케일링이 들어가야 함. 현재는 round만 수행
        x = self.quantizer(x) 
        return x

class SplitSender(nn.Module):
    def __init__(self, n_channels=3):
        super().__init__()
        
        # Encoder Path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Compressors (압축률 조정 가능)
        self.cmp_skip1 = Compressor(64, 8)     # 1/8
        self.cmp_skip2 = Compressor(128, 16)   # 1/8
        self.cmp_skip3 = Compressor(256, 32)   # 1/8
        self.cmp_bot   = Compressor(1024, 128) # 1/8

    def forward(self, x):
        # Encoding
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        # Compression
        packet_skip1 = self.cmp_skip1(x1)
        packet_skip2 = self.cmp_skip2(x2)
        packet_skip3 = self.cmp_skip3(x3)
        packet_bot   = self.cmp_bot(x5)

        # Return Packets
        return packet_bot, [packet_skip3, packet_skip2, packet_skip1]