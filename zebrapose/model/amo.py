import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import ResNet34_OS8


class AMO(nn.Module):
    def __init__(self):
        # this module's input is img, mask, and
        super(AMO, self).__init__()
        self.resnet = ResNet34_OS8(34, concat_decoder=True)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(65, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, 32, 1, 1)
        )
        self.upsample_1 = self.upsample(32, 32, 3, padding=1, output_padding=1)

    def upsample(self, in_channels, num_filters, kernel_size, padding, output_padding):
        upsample_layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels,
                                num_filters,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=padding,
                                output_padding=output_padding,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True)
                        )
        return upsample_layer

    def forward(self, x, mask):
        x_high_feature, x_128, x_64, x_32, x_16 = self.resnet(x)
        fm = self.conv1(torch.cat[x_high_feature, x_32])
        fm_1 = self.conv_1(x)  # 【bsz, 3, 256, 256] -> [bsz, 32, 128, 128]
        x
        fm = self.conv_1(torch.cat([fm, mask, x_128], 1))  # 【bsz, 97, 128, 128] -> [bsz, 1, 128, 128]
