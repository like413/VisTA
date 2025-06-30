import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from model.decoder import Reason_Decoder, Mask_Decoder
from model.layers import FPN


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class VisTA(nn.Module):
    def __init__(self):
        super(VisTA, self).__init__()
        clip_model = torch.jit.load('pretrain/RN101.pt', map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), 40).float()
        self.neck = FPN(in_channels=[512, 1024, 512], out_channels=[256, 512, 1024])

        self.ReasonDecoder = Reason_Decoder()

        self.MaskDecoder = Mask_Decoder()

        self.down1 = Conv(1024, 512)
        self.down2 = Conv(2048, 1024)
        self.down3 = Conv(1024, 512)

    def forward(self, img1, img2, word, mask=None, answer_vec=None):
        vis1 = self.backbone.encode_image(img1)
        vis2 = self.backbone.encode_image(img2)
        word, state = self.backbone.encode_text(word)

        v1 = self.down1(torch.cat([vis1[0], vis2[0]], dim=1))
        v2 = self.down2(torch.cat([vis1[1], vis2[1]], dim=1))
        v3 = self.down3(torch.cat([vis1[2], vis2[2]], dim=1))

        fv = self.neck([v1, v2, v3], state)

        mask_temp, ans, src = self.ReasonDecoder(fv, word, state)

        pred = self.MaskDecoder(vis=fv, masks=mask_temp)

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
            loss1 = nn.CrossEntropyLoss()(ans.float(), answer_vec)
            loss2 = F.binary_cross_entropy_with_logits(pred.float(), mask)

            loss = loss1 * 0.2 + loss2
            return pred.detach(), ans.detach(), mask, loss, loss1, loss2
        else:
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != mask.shape[-2:]:
                pred = F.interpolate(pred, size=mask.shape[-2:], mode='bicubic', align_corners=True).squeeze(1)
            return pred.detach(), ans.detach()
