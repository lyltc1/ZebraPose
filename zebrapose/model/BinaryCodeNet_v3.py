import torch.nn as nn
import torch
import torch.nn.functional as F
from model.resnet import ResNet50_OS8, ResNet34_OS8


#################### loss ######################
################################################
class BinaryCodeLoss(nn.Module):
    def __init__(self, binary_code_loss_type, mask_binary_code_loss, divided_number_each_iteration,
                 use_histgramm_weighted_binary_loss=False):
        super().__init__()
        self.binary_code_loss_type = binary_code_loss_type
        self.mask_binary_code_loss = mask_binary_code_loss
        self.divided_number_each_iteration = divided_number_each_iteration
        self.use_histgramm_weighted_binary_loss = use_histgramm_weighted_binary_loss

        if self.use_histgramm_weighted_binary_loss:  # this Hammloss will be used in both case, for loss, or for histogramm
            self.hamming_loss = HammingLoss()

        if binary_code_loss_type == "L1":
            self.loss = nn.L1Loss(reduction="mean")
        elif binary_code_loss_type == "BCE":
            self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        elif binary_code_loss_type == "CE":
            self.loss = nn.CrossEntropyLoss(reduction="mean")
        else:
            raise NotImplementedError(f"unknown mask loss type: {binary_code_loss_type}")

        if self.use_histgramm_weighted_binary_loss:
            assert binary_code_loss_type == "BCE"  # currently only have the implementation with BCEWithLogitsLoss
            self.loss = BinaryLossWeighted(nn.BCEWithLogitsLoss(reduction='none'))

        self.histogram = None

    def forward(self, pred_binary_code, pred_mask, groundtruth_code):
        ## calculating hamming loss and bit error histogram for loss weighting
        if self.use_histgramm_weighted_binary_loss:
            loss_hamm, histogram_new = self.hamming_loss(pred_binary_code, groundtruth_code, pred_mask.clone().detach())
            if self.histogram is None:
                self.histogram = histogram_new
            else:
                self.histogram = histogram_new * 0.05 + self.histogram * 0.95

            ## soft bin weigt decrease 
            hist_soft = torch.minimum(self.histogram, 0.51 - self.histogram)
            bin_weights = torch.exp(hist_soft * 3).clone()

        if self.mask_binary_code_loss:
            pred_binary_code = pred_mask.clone().detach() * pred_binary_code

        if self.binary_code_loss_type == "L1":
            pred_binary_code = pred_binary_code.reshape(-1, 1, pred_binary_code.shape[2], pred_binary_code.shape[3])
            pred_binary_code = torch.sigmoid(pred_binary_code)
            groundtruth_code = groundtruth_code.view(-1, 1, groundtruth_code.shape[2], groundtruth_code.shape[3])
        elif self.binary_code_loss_type == "BCE" and not self.use_histgramm_weighted_binary_loss:
            pred_binary_code = pred_binary_code.reshape(-1, pred_binary_code.shape[2], pred_binary_code.shape[3])
            groundtruth_code = groundtruth_code.view(-1, groundtruth_code.shape[2], groundtruth_code.shape[3])
        elif self.binary_code_loss_type == "CE":
            pred_binary_code = pred_binary_code.reshape(-1, self.divided_number_each_iteration,
                                                        pred_binary_code.shape[2], pred_binary_code.shape[3])
            groundtruth_code = groundtruth_code.view(-1, groundtruth_code.shape[2], groundtruth_code.shape[3])
            groundtruth_code = groundtruth_code.long()

        if self.use_histgramm_weighted_binary_loss:
            loss = self.loss(pred_binary_code, groundtruth_code, bin_weights)
        else:
            loss = self.loss(pred_binary_code, groundtruth_code)

        return loss


class BinaryLossWeighted(nn.Module):
    def __init__(self, baseloss):
        # the base loss should have reduction 'none'
        super().__init__()
        self.base_loss = baseloss

    def forward(self, input, target, weight):
        base_output = self.base_loss(input, target)
        assert base_output.ndim == 4
        output = base_output.mean([0, 2, 3])
        output = torch.sum(output * weight) / torch.sum(weight)
        return output


class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="mean")

    def forward(self, pred_mask, groundtruth_mask):
        pred_mask = pred_mask[:, 0, :, :]
        pred_mask = torch.sigmoid(pred_mask)

        return self.loss(pred_mask, groundtruth_mask)


class HammingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted_code_prob, GT_code, mask):
        assert predicted_code_prob.ndim == 4
        mask_hard = mask.round().clamp(0, 1)  # still kept round and clamp for safety
        code1_hard = torch.sigmoid(predicted_code_prob).round().clamp(0, 1)
        code2_hard = GT_code.round().clamp(0, 1)  # still kept round and clamp for safety
        hamm = torch.abs(code1_hard - code2_hard) * mask_hard
        histogram = hamm.sum([0, 2, 3]) / (mask_hard.sum() + 1)
        hamm_loss = histogram.mean()

        return hamm_loss, histogram.detach()


class BinaryCodeNet_Deeplab_v3(nn.Module):
    def __init__(
            self,
            num_resnet_layers,
            binary_code_length,
            divided_number_each_iteration,
            concat=False,
            output_kernel_size=1,
    ):
        super(BinaryCodeNet_Deeplab_v3, self).__init__()
        self.concat = concat

        if divided_number_each_iteration == 2:
            # hard coded 1 for object mask
            self.net = DeepLabV3(num_resnet_layers, binary_code_length + 1 + 1, concat=self.concat,
                                 output_kernel_size=output_kernel_size)

    def forward(self, inputs):
        return self.net(inputs)


class DeepLabV3(nn.Module):
    def __init__(self, num_resnet_layers, num_classes, concat=False, output_kernel_size=1):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes
        self.concat = concat
        self.num_resnet_layers = num_resnet_layers
        self.resnet = ResNet34_OS8(34, concat)  # NOTE! specify the type of ResNet here
        self.aspp = ASPP(num_classes=self.num_classes, concat=concat, output_kernel_size=output_kernel_size)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))
        # [bsz,512,32,32],[bsz,64,128,128],[bsz,64,64,64],[bsz,128,32,32],[bsz,256,32,32]
        x_high_feature, x_128, x_64, x_32, x_16 = self.resnet(x)
        output = self.aspp(x_high_feature, x_128, x_64, x_32, x_16)  #[bsz,1+code,128,128]
        mask, entire_mask, binary_code = torch.split(output, [1, 1, self.num_classes - 2], 1)

        return mask, entire_mask, binary_code


class ASPP(nn.Module):
    def __init__(self, num_classes, concat=True, output_kernel_size=1):
        super(ASPP, self).__init__()
        self.concat = concat

        #####ASPP
        self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        ############

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)  # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        #####start upsample here
        kernel_size = 3
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        if self.concat:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding)  # from 32x32 to 64x64
            self.upsample_2 = self.upsample(256 + 64, 256, 3, padding, output_padding)  # from 64x64 to 128x128

        else:
            self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding)
            self.upsample_2 = self.upsample(256, 256, 3, padding, output_padding)

        if output_kernel_size == 3:
            padding = 1
        elif output_kernel_size == 2:
            padding = 0
        elif output_kernel_size == 1:
            padding = 0

        self.conv_1x1_4 = nn.Conv2d(256 + 64, num_classes, kernel_size=output_kernel_size, padding=padding)

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

    def forward(self, x_high_feature, x_128=None, x_64=None, x_32=None, x_16=None):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = x_high_feature.size()[2]  # (== h/16 or h/8)
        feature_map_w = x_high_feature.size()[3]  # (== w/16 or w/8)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x_high_feature)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(
            self.bn_conv_3x3_1(self.conv_3x3_1(x_high_feature)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(
            self.bn_conv_3x3_2(self.conv_3x3_2(x_high_feature)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(
            self.bn_conv_3x3_3(self.conv_3x3_3(x_high_feature)))  # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(x_high_feature)  # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))  # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w),
                                mode="bilinear")  # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img],
                        1)  # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))  # (shape: (batch_size, 256, h/16, w/16))

        # need 3 times deconv, 16 -> 32, 32 -> 64, 64->128
        if self.concat:
            x = self.upsample_1(out)

            x = torch.cat([x, x_64], 1)
            x = self.upsample_2(x)

        else:
            x = self.upsample_1(out)
            x = self.upsample_2(x)

        x = self.conv_1x1_4(torch.cat([x, x_128], 1))  # (shape: (batch_size, num_classes, h/16, w/16))

        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()

    def forward(self, x, mask, entire_mask, x_high_feature, x_128=None, x_64=None, x_32=None, x_16=None):
        feature_map_h = x_high_feature.size()[2]
        feature_map_w = x_high_feature.size()[3]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x_high_feature)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(
            self.bn_conv_3x3_1(self.conv_3x3_1(x_high_feature)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(
            self.bn_conv_3x3_2(self.conv_3x3_2(x_high_feature)))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(
            self.bn_conv_3x3_3(self.conv_3x3_3(x_high_feature)))  # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(x_high_feature)  # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))  # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w),
                                mode="bilinear")  # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img],
                        1)  # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))  # (shape: (batch_size, 256, h/16, w/16))

        # need 3 times deconv, 16 -> 32, 32 -> 64, 64->128
        if self.concat:
            x = self.upsample_1(out)

            x = torch.cat([x, x_64], 1)
            x = self.upsample_2(x)

        else:
            x = self.upsample_1(out)
            x = self.upsample_2(x)

        x = self.conv_1x1_4(torch.cat([x, x_128], 1))  # (shape: (batch_size, num_classes, h/16, w/16))

        return x