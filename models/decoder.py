import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) * 2 (Decoder용)"""
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

class SplitReceiver(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()

        # Decompressors (채널 복원: 1x1 Conv)
        self.dec_bot = nn.Conv2d(128, 1024, kernel_size=1)
        self.dec_skip3 = nn.Conv2d(32, 256, kernel_size=1)
        self.dec_skip2 = nn.Conv2d(16, 128, kernel_size=1)
        self.dec_skip1 = nn.Conv2d(8, 64, kernel_size=1)

        # Decoder Path
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512 + 256, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256 + 128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128 + 64, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.final_act = nn.Sigmoid() # 화면 출력용 (0~1 범위)

    def forward(self, compressed_bot, compressed_skips):
        p_sk3, p_sk2, p_sk1 = compressed_skips

        # 1. Decompression
        x5 = self.dec_bot(compressed_bot)
        sk3 = self.dec_skip3(p_sk3)
        sk2 = self.dec_skip2(p_sk2)
        sk1 = self.dec_skip1(p_sk1)

        # 2. Decoding & Concat
        x = self.up1(x5)
        x = self._concat(x, sk3)
        x = self.conv1(x)

        x = self.up2(x)
        x = self._concat(x, sk2)
        x = self.conv2(x)

        x = self.up3(x)
        x = self._concat(x, sk1)
        x = self.conv3(x)

        x = self.up4(x)
        logits = self.outc(x)
        
        return self.final_act(logits)

    def _concat(self, upsampled, skip):
        # 패딩을 통해 크기 불일치 해결
        diffY = skip.size()[2] - upsampled.size()[2]
        diffX = skip.size()[3] - upsampled.size()[3]
        upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        return torch.cat([skip, upsampled], dim=1)