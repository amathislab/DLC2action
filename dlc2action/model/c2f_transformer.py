#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import math
from functools import partial
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dlc2action.model.base_model import Model

nonlinearity = partial(F.relu, inplace=True)


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, fc=False):
        super(double_conv, self).__init__()
        if fc:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 5
            padding = 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass."""
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        """Forward pass."""
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        """Forward pass."""
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(nn.MaxPool1d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        """Forward pass."""
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, in_channels, out_channels, heads, att_in, bilinear=True, fc=False
    ):
        super().__init__()

        self.attn = MultiHeadAttention(heads=heads, d_model=att_in)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.out = double_conv(in_channels, out_channels, fc=fc)

    def forward(self, x1, x2):
        """Forward pass."""
        x1 = self.attn(x1, x1, x1)
        x1 = self.up(x1)
        diff = torch.tensor([x2.size()[2] - x1.size()[2]])

        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.out(x)


class TPPblock(nn.Module):
    def __init__(self, in_channels):
        super(TPPblock, self).__init__()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool4 = nn.MaxPool1d(kernel_size=6, stride=6)

        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=1, kernel_size=1, padding=0
        )

        self.out_conv = nn.Conv1d(
            in_channels=in_channels + 4,
            out_channels=in_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x):
        """Forward pass."""
        self.in_channels, t = x.size(1), x.size(2)
        self.layer1 = F.interpolate(
            self.conv(self.pool1(x)), size=t, mode="linear", align_corners=True
        )
        self.layer2 = F.interpolate(
            self.conv(self.pool2(x)), size=t, mode="linear", align_corners=True
        )
        self.layer3 = F.interpolate(
            self.conv(self.pool3(x)), size=t, mode="linear", align_corners=True
        )
        self.layer4 = F.interpolate(
            self.conv(self.pool4(x)), size=t, mode="linear", align_corners=True
        )

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)
        out = self.out_conv(out)

        return out


class C2F_Transformer_Module(nn.Module):
    """
    Features are extracted at the last layer of decoder.
    """

    def __init__(
        self, n_channels, output_dim, heads, num_f_maps, use_predictor=False, fc=False
    ):
        super().__init__()
        self.use_predictor = use_predictor
        self.inc = inconv(n_channels, num_f_maps * 2)
        self.down1 = down(num_f_maps * 2, num_f_maps * 2)
        self.down2 = down(num_f_maps * 2, num_f_maps * 2)
        self.down3 = down(num_f_maps * 2, num_f_maps)
        self.down4 = down(num_f_maps, num_f_maps)
        self.down5 = down(num_f_maps, num_f_maps)
        self.down6 = down(num_f_maps, num_f_maps)
        self.pe = PositionalEncoder(num_f_maps)
        self.up = up(num_f_maps * 2, num_f_maps, heads, att_in=num_f_maps, fc=fc)
        self.outcc0 = outconv(num_f_maps, output_dim)
        self.up0 = up(num_f_maps * 2, num_f_maps, heads, att_in=num_f_maps, fc=fc)
        self.outcc1 = outconv(num_f_maps, output_dim)
        self.up1 = up(num_f_maps * 2, num_f_maps, heads, att_in=num_f_maps, fc=fc)
        self.outcc2 = outconv(num_f_maps, output_dim)
        self.up2 = up(num_f_maps * 3, num_f_maps, heads, att_in=num_f_maps, fc=fc)
        self.outcc3 = outconv(num_f_maps, output_dim)
        self.up3 = up(num_f_maps * 3, num_f_maps, heads, att_in=num_f_maps, fc=fc)
        self.outcc4 = outconv(num_f_maps, output_dim)
        self.up4 = up(num_f_maps * 3, num_f_maps, heads, att_in=num_f_maps, fc=fc)
        self.outcc = outconv(num_f_maps, output_dim)
        self.tpp = TPPblock(num_f_maps)
        self.weights = torch.nn.Parameter(torch.ones(6))

    def forward(self, x):
        """Forward pass."""
        # print(f'{x.shape=}')
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x7 = self.tpp(x7)
        x7 = self.pe(x7)
        # print(f'{x6.shape=}')
        x = self.up(x7, x6)
        y1 = self.outcc0(F.relu(x))
        x = self.up0(x, x5)
        y2 = self.outcc1(F.relu(x))
        x = self.up1(x, x4)
        y3 = self.outcc2(F.relu(x))
        x = self.up2(x, x3)
        y4 = self.outcc3(F.relu(x))
        x = self.up3(x, x2)
        y5 = self.outcc4(F.relu(x))
        x = self.up4(x, x1)
        y = self.outcc(x)
        output = [y]
        for outp_ele in [y5, y4, y3]:
            output.append(
                F.interpolate(
                    outp_ele, size=y.shape[-1], mode="linear", align_corners=True
                )
            )
        output = torch.stack(output, dim=0)
        if self.use_predictor:
            K, B, C, T = output.shape
            output = output.reshape((-1, C, T))
        return output


class C2F_Transformer(Model):
    """
    A modification of C2F-TCN that replaces some convolutions with attention

    Requires the `"general/len_segment"` parameter to be at least 512
    """

    def __init__(
        self,
        num_classes,
        input_dims,
        heads,
        num_f_maps,
        linear=False,
        feature_dim=None,
        state_dict_path=None,
        ssl_constructors=None,
        ssl_types=None,
        ssl_modules=None,
    ):
        input_dims = int(sum([s[0] for s in input_dims.values()]))
        if feature_dim is None:
            feature_dim = num_classes
            self.f_shape = None
            self.params_predictor = None
        else:
            self.f_shape = torch.Size([feature_dim])
            self.params_predictor = {
                "dim": int(feature_dim),
                "num_classes": num_classes,
            }
        self.params = {
            "output_dim": int(float(feature_dim)),
            "n_channels": int(float(input_dims)),
            "num_f_maps": int(float(num_f_maps)),
            "heads": int(float(heads)),
            "use_predictor": self.f_shape is not None,
            "fc": linear,
        }
        super().__init__(ssl_constructors, ssl_modules, ssl_types, state_dict_path)

    def _feature_extractor(self) -> Union[torch.nn.Module, List]:
        return C2F_Transformer_Module(**self.params)

    def _predictor(self) -> torch.nn.Module:
        if self.params_predictor is not None:
            return Predictor(**self.params_predictor)
        else:
            return nn.Identity()

    def features_shape(self) -> Optional[torch.Size]:
        return self.f_shape


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependent on
        # pos and i
        pe = torch.zeros(d_model, max_seq_len)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[i, pos] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pe[i + 1, pos] = math.cos(
                        pos / (10000 ** ((2 * (i + 1)) / d_model))
                    )

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        """Forward pass."""
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(-1)
        x = x + self.pe[:, :, :seq_len].to(x.device)
        return x


class Predictor(nn.Module):
    def __init__(self, dim, num_classes):
        super(Predictor, self).__init__()
        self.num_classes = num_classes
        self.conv_out_1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.conv_out_2 = nn.Conv1d(dim, num_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass."""
        x = self.conv_out_1(x)
        x = F.relu(x)
        x = self.conv_out_2(x)
        x = x.reshape((4, -1, self.num_classes, x.shape[-1]))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        # print(f'{d_model=}')
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """Forward pass."""
        bs = q.size(0)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2)

        # perform linear operation and split into h heads
        # print(f'{self.h=}, {self.d_k=}, {k.shape=}')
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat).transpose(1, 2)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    """Attention."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output
