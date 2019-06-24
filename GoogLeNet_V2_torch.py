import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch


class _BasicConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, **kwargs):
        super(_BasicConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(num_features=channels_out, eps=1e-3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn2(x)
        # x = nn.ReLU(inplace=False)(x)
        # return x
        return F.relu(x, inplace=False)




class _Inception_V2(nn.Module):
    def __init__(self, channels_in, channels_1x1_out, channels_3x3_reduce_out, channels_3x3_out,
                 channels_double_3x3_reduce_out, channels_double_3x3_a_out, channels_double_3x3_b_out,
                 channels_pool_proj):
        super(_Inception_V2, self).__init__()
        self.conv1x1 = _BasicConv2d(channels_in=channels_in, channels_out=channels_1x1_out, kernel_size=1, stride=1)

        self.conv3x3_reduce = _BasicConv2d(channels_in, channels_3x3_reduce_out, kernel_size=1, stride=1)
        self.conv3x3 = _BasicConv2d(channels_3x3_reduce_out, channels_3x3_out, kernel_size=3, stride=1, padding=1)

        self.conv_double3x3_reduce = _BasicConv2d(channels_in, channels_double_3x3_reduce_out, kernel_size=1, stride=1)
        self.conv_double3x3_a = _BasicConv2d(channels_double_3x3_reduce_out, channels_double_3x3_a_out, kernel_size=3, stride=1, padding=1)
        self.conv_double3x3_b = _BasicConv2d(channels_double_3x3_a_out, channels_double_3x3_b_out, kernel_size=3, stride=1, padding=1)

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = _BasicConv2d(channels_in, channels_pool_proj, kernel_size=1, stride=1)


    def forward(self, x):
        x_1x1 = self.conv1x1(x)

        x_3x3 = self.conv3x3_reduce(x)
        x_3x3 = self.conv3x3(x_3x3)

        x_double3x3 = self.conv_double3x3_reduce(x)
        x_double3x3 = self.conv_double3x3_a(x_double3x3)
        x_double3x3 = self.conv_double3x3_b(x_double3x3)

        # x_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(x)
        # x_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)
        x_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_pool = self.pool_conv(x_pool)

        x_out = [x_1x1, x_3x3, x_double3x3, x_pool]
        x_out = torch.cat(x_out, dim=1)
        return x_out


class _Inception_V2_downsample(nn.Module):
    def __init__(self, channels_in, channels_3x3_reduce_out, channels_3x3_out, channels_double_3x3_reduce_out,
                 channels_double_3x3_a_out, channels_double_3x3_b_out):
        super(_Inception_V2_downsample, self).__init__()

        self.conv3x3_reduce = _BasicConv2d(channels_in, channels_3x3_reduce_out, kernel_size=1, stride=1)
        self.conv3x3 = _BasicConv2d(channels_3x3_reduce_out, channels_3x3_out, kernel_size=3, stride=2, padding=1)

        self.conv_double3x3_reduce = _BasicConv2d(channels_in, channels_double_3x3_reduce_out, kernel_size=1, stride=1)
        self.conv_double3x3_a = _BasicConv2d(channels_double_3x3_reduce_out, channels_double_3x3_a_out, kernel_size=3, stride=1, padding=1)
        self.conv_double3x3_b = _BasicConv2d(channels_double_3x3_a_out, channels_double_3x3_b_out, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x1 = self.conv3x3_reduce(x)
        x1 = self.conv3x3(x1)

        x2 = self.conv_double3x3_reduce(x)
        x2 = self.conv_double3x3_a(x2)
        x2 = self.conv_double3x3_b(x2)

        x3 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        out = [x1, x2, x3]

        return torch.cat(out, dim=1)







class _GoogLeNetV2(nn.Module):
    def __init__(self, nclasses=1000):
        super(_GoogLeNetV2, self).__init__()
        # self.H_size = H_size
        # self.W_size = W_size
        # self.channels = channels
        # self.nclasses = nclasses

        self.conv1_7x7_s2 = _BasicConv2d(channels_in=3, channels_out=64, kernel_size=7, stride=2, padding=3)

        self.conv2_3x3_reduce = _BasicConv2d(64, 64, kernel_size=1, stride=1)
        self.conv2_3x3 = _BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)

        self._inception_3a = _Inception_V2(channels_in=192, channels_1x1_out=64,
                                           channels_3x3_reduce_out=64, channels_3x3_out=64,
                                           channels_double_3x3_reduce_out=64,
                                           channels_double_3x3_a_out=96, channels_double_3x3_b_out=96,
                                           channels_pool_proj=32)
        self._inception_3b = _Inception_V2(channels_in=256, channels_1x1_out=64,
                                           channels_3x3_reduce_out=64, channels_3x3_out=96,
                                           channels_double_3x3_reduce_out=64,
                                           channels_double_3x3_a_out=96, channels_double_3x3_b_out=96,
                                           channels_pool_proj=64)
        self._inception_3c = _Inception_V2_downsample(channels_in=320, channels_3x3_reduce_out=128,
                                                      channels_3x3_out=160, channels_double_3x3_reduce_out=64,
                                                      channels_double_3x3_a_out=96, channels_double_3x3_b_out=96)

        self._inception_4a = _Inception_V2(channels_in=576, channels_1x1_out=224,
                                           channels_3x3_reduce_out=64, channels_3x3_out=96,
                                           channels_double_3x3_reduce_out=96, channels_double_3x3_a_out=128,
                                           channels_double_3x3_b_out=128, channels_pool_proj=128)
        self._inception_4b = _Inception_V2(channels_in=576, channels_1x1_out=192,
                                           channels_3x3_reduce_out=96, channels_3x3_out=128,
                                           channels_double_3x3_reduce_out=96, channels_double_3x3_a_out=128,
                                           channels_double_3x3_b_out=128, channels_pool_proj=128)
        self._inception_4c = _Inception_V2(channels_in=576, channels_1x1_out=160,
                                           channels_3x3_reduce_out=128, channels_3x3_out=160,
                                           channels_double_3x3_reduce_out=128, channels_double_3x3_a_out=160,
                                           channels_double_3x3_b_out=160, channels_pool_proj=96)
        self._inception_4d = _Inception_V2(channels_in=576, channels_1x1_out=96,
                                           channels_3x3_reduce_out=128, channels_3x3_out=192,
                                           channels_double_3x3_reduce_out=160, channels_double_3x3_a_out=192,
                                           channels_double_3x3_b_out=192, channels_pool_proj=96)
        self._inception_4e = _Inception_V2_downsample(channels_in=576, channels_3x3_reduce_out=128,
                                                      channels_3x3_out=192, channels_double_3x3_reduce_out=192,
                                                      channels_double_3x3_a_out=256, channels_double_3x3_b_out=256)

        self._inception_5a = _Inception_V2(channels_in=1024, channels_1x1_out=352,
                                           channels_3x3_reduce_out=192, channels_3x3_out=320,
                                           channels_double_3x3_reduce_out=160, channels_double_3x3_a_out=224,
                                           channels_double_3x3_b_out=224, channels_pool_proj=128)
        self._inception_5b = _Inception_V2(channels_in=1024, channels_1x1_out=352,
                                           channels_3x3_reduce_out=192, channels_3x3_out=320,
                                           channels_double_3x3_reduce_out=192, channels_double_3x3_a_out=224,
                                           channels_double_3x3_b_out=224, channels_pool_proj=128)

        self.classifier = nn.Linear(1024, nclasses)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight.data, a=0, mode="fan_in")


    def forward(self, x):
        x = self.conv1_7x7_s2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.conv2_3x3_reduce(x)
        x = self.conv2_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self._inception_3a(x)
        x = self._inception_3b(x)
        x = self._inception_3c(x)
        # print("inception3:", x.size())

        x = self._inception_4a(x)
        x = self._inception_4b(x)
        x = self._inception_4c(x)
        x = self._inception_4d(x)
        x = self._inception_4e(x)

        x = self._inception_5a(x)
        x = self._inception_5b(x)

        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.squeeze()
        # print("avgpool: ", x.size())

        x = self.classifier(x)
        return x



if __name__ == '__main__':
    googlenet_frame_v2 = _GoogLeNetV2(nclasses=100)
    print(googlenet_frame_v2)

    # googlenet_frame_v2.train()

    del googlenet_frame_v2
    print("done")



