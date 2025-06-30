# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from typing import Type

import math
import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class ReasonDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.project = Projector(512, 512 // 2, 3)

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(
            self,
            image_embeddings: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            state,
            multimask_output: bool,
    ):
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
        """
        B, C, H, W = image_embeddings.size()
        image_pe = self.pos2d(C, H, W).to(image_embeddings.device)
        masks, ans, src = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            state=state,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]

        # Prepare output
        return masks, ans, src

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            state
    ):
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = image_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        masks, ans = self.project(src, state)

        return masks, ans, src


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ln_ans = nn.Conv2d(512, 512, 1)
        self.ln_ans2 = nn.Conv2d(512, 128, 1)
        self.ln = nn.Linear(128, 23)

    def forward(self, x, word):
        ans = self.global_avgpool(x)
        ans_vec = self.ln_ans(ans)

        x = self.vis(x)
        B, C, H, W = x.size()
        x = x.reshape(1, B * C, H, W)
        qa = torch.cat([word, torch.flatten(ans_vec, 1)], dim=1)

        out = F.softmax(qa, dim=1) * qa
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)

        word = out1 + out2  # Q+A

        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        out = out.transpose(0, 1)
        ans = self.ln(torch.flatten(self.ln_ans2(ans_vec), 1))
        return out, ans
