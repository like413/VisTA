import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class FPN(nn.Module):
    def __init__(self, in_channels=[512, 1024, 1024], out_channels=[256, 512, 1024]):
        super(FPN, self).__init__()
        # text projection
        self.txt_proj = linear_layer(in_channels[2], out_channels[2])
        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))

        self.upConv = nn.ConvTranspose2d(out_channels[2], out_channels[2], 2, 2)
        self.upConv2 = nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, 2)

    def forward(self, imgs, state):
        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = imgs
        # fusion 1: b, 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(
            -1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        f5 = self.norm_layer(f5 * state)
        # fusion 2: b, 512, 26, 26
        f4 = self.f2_v_proj(v4)
        # f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')
        f5_ = self.upConv(f5)
        f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
        # fusion 3: b, 256, 26, 26
        f3 = self.f3_v_proj(v3)
        #
        f4_ = self.upConv2(f4)

        f3 = self.f3_cat(torch.cat([f3, f4_], dim=1))
        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        fq3 = F.avg_pool2d(fq3, 2, 2)
        # query
        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        fq = self.coordconv(fq)
        # b, 512, 26, 26
        return fq
