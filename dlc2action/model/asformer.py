#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
# Incorporates code adapted from ASFormer by ChinaYi
# Original work Copyright (c) 2021 ChinaYi
# Source: https://github.com/ChinaYi/ASFormer
# Originally licensed under MIT License
# Combined work licensed under GNU AGPLv3
#
import copy
import math
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dlc2action.model.base_model import Model
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exponential_descrease(idx_decoder, p=3):
    """Exponential decrease function for the attention window."""
    return math.exp(-p * idx_decoder)


class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        """
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        """
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape

        assert c1 == c2

        energy = torch.bmm(
            proj_query.permute(0, 2, 1), proj_key
        )  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(
            padding_mask + 1e-6
        )  # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention)
        attention = attention * padding_mask
        attention = attention.permute(0, 2, 1)
        out = torch.bmm(proj_val, attention)
        return out, attention


class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type):  # r1 = r2
        self._fix_types(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)
        super(AttLayer, self).__init__()

        self.query_conv = nn.Conv1d(
            in_channels=self.q_dim, out_channels=self.q_dim // self.r1, kernel_size=1
        )
        self.key_conv = nn.Conv1d(
            in_channels=self.k_dim, out_channels=self.k_dim // self.r2, kernel_size=1
        )
        self.value_conv = nn.Conv1d(
            in_channels=self.v_dim, out_channels=self.v_dim // self.r3, kernel_size=1
        )

        self.conv_out = nn.Conv1d(
            in_channels=self.v_dim // self.r3, out_channels=self.v_dim, kernel_size=1
        )

        assert self.att_type in ["normal_att", "block_att", "sliding_att"]
        assert self.stage in ["encoder", "decoder"]

        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()

    def _fix_types(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type):
        self.q_dim = int(float(q_dim))
        self.k_dim = int(float(k_dim))
        self.v_dim = int(float(v_dim))
        self.r1 = int(float(r1))
        self.r2 = int(float(r2))
        self.r3 = int(float(r3))
        self.bl = int(float(bl))
        self.stage = stage
        self.att_type = att_type

    def construct_window_mask(self):
        """
        construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        """
        window_mask = torch.zeros((1, self.bl, self.bl + 2 * (self.bl // 2)))
        for i in range(self.bl):
            window_mask[:, :, i : i + self.bl] = 1
        return window_mask.to(device)

    def forward(self, x1, x2, mask):
        """Forward pass."""
        # x1 from the encoder
        # x2 from the decoder

        query = self.query_conv(x1)
        key = self.key_conv(x1)

        if self.stage == "decoder":
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)

        if self.att_type == "normal_att":
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == "block_att":
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == "sliding_att":
            return self._sliding_window_self_att(query, key, value, mask)

    def _normal_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _block_wise_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()

        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat(
                [q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            k = torch.cat(
                [k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            v = torch.cat(
                [v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            nb += 1

        padding_mask = torch.cat(
            [
                torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :],
                torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device),
            ],
            dim=-1,
        )

        q = (
            q.reshape(m_batchsize, c1, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, c1, self.bl)
        )
        padding_mask = (
            padding_mask.reshape(m_batchsize, 1, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, 1, self.bl)
        )
        k = (
            k.reshape(m_batchsize, c2, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, c2, self.bl)
        )
        v = (
            v.reshape(m_batchsize, c3, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, c3, self.bl)
        )

        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))

        output = (
            output.reshape(m_batchsize, nb, c3, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize, c3, nb * self.bl)
        )
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]

    def _sliding_window_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()

        # padding zeros for the last segment
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat(
                [q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            k = torch.cat(
                [k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            v = torch.cat(
                [v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)],
                dim=-1,
            )
            nb += 1
        padding_mask = torch.cat(
            [
                torch.ones((m_batchsize, 1, L)).to(device) * mask[:, 0:1, :],
                torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device),
            ],
            dim=-1,
        )

        # sliding window approach, by splitting query_proj and key_proj into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = (
            q.reshape(m_batchsize, c1, nb, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize * nb, c1, self.bl)
        )

        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat(
            [
                torch.zeros(m_batchsize, c2, self.bl // 2).to(device),
                k,
                torch.zeros(m_batchsize, c2, self.bl // 2).to(device),
            ],
            dim=-1,
        )
        v = torch.cat(
            [
                torch.zeros(m_batchsize, c3, self.bl // 2).to(device),
                v,
                torch.zeros(m_batchsize, c3, self.bl // 2).to(device),
            ],
            dim=-1,
        )
        padding_mask = torch.cat(
            [
                torch.zeros(m_batchsize, 1, self.bl // 2).to(device),
                padding_mask,
                torch.zeros(m_batchsize, 1, self.bl // 2).to(device),
            ],
            dim=-1,
        )

        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat(
            [
                k[:, :, i * self.bl : (i + 1) * self.bl + (self.bl // 2) * 2]
                for i in range(nb)
            ],
            dim=0,
        )  # special case when self.bl = 1
        v = torch.cat(
            [
                v[:, :, i * self.bl : (i + 1) * self.bl + (self.bl // 2) * 2]
                for i in range(nb)
            ],
            dim=0,
        )
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat(
            [
                padding_mask[:, :, i * self.bl : (i + 1) * self.bl + (self.bl // 2) * 2]
                for i in range(nb)
            ],
            dim=0,
        )  # of shape (m*nb, 1, 2l)
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * padding_mask

        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))

        output = (
            output.reshape(m_batchsize, nb, -1, self.bl)
            .permute(0, 2, 1, 3)
            .reshape(m_batchsize, -1, nb * self.bl)
        )
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
        #         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(
                    AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)
                )
                for i in range(num_head)
            ]
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x1, x2, mask):
        """Forward pass."""
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out


class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, 3, padding=dilation, dilation=dilation
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        """Forward pass."""
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1),
        )

    def forward(self, x):
        """Forward pass."""
        return self.layer(x)


class AttModule(nn.Module):
    def __init__(
        self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha
    ):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(
            in_channels,
            in_channels,
            out_channels,
            r1,
            r1,
            r2,
            dilation,
            att_type=att_type,
            stage=stage,
        )  # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        """Forward pass."""
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)  # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)

    #         self.register_buffer('pe', pe)

    def forward(self, x):
        """Forward pass."""
        return x + self.pe[:, :, 0 : x.shape[2]]


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        r1,
        r2,
        num_f_maps,
        input_dim,
        channel_masking_rate,
        att_type,
        alpha,
    ):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)  # fc layer
        self.layers = nn.ModuleList(
            [
                AttModule(
                    2**i, num_f_maps, num_f_maps, r1, r2, att_type, "encoder", alpha
                )
                for i in range(num_layers)  # 2**i
            ]
        )

        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        """Forward pass."""
        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)

        return feature


class Decoder(nn.Module):
    def __init__(
        self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha
    ):
        super(
            Decoder, self
        ).__init__()  # self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                AttModule(
                    2**i, num_f_maps, num_f_maps, r1, r2, att_type, "decoder", alpha
                )
                for i in range(num_layers)  # 2 ** i
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):
        """Forward pass."""
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class MyTransformer(nn.Module):
    def __init__(
        self,
        num_layers,
        r1,
        r2,
        num_f_maps,
        input_dim,
        channel_masking_rate,
    ):
        super(MyTransformer, self).__init__()
        self.encoder = Encoder(
            num_layers,
            r1,
            r2,
            num_f_maps,
            input_dim,
            channel_masking_rate,
            att_type="sliding_att",
            alpha=1,
        )

    def forward(self, x):
        """Forward pass."""
        mask = (x.sum(1).unsqueeze(1) != 0).int()
        feature = self.encoder(x, mask)
        feature = feature * mask

        return feature


class Predictor(nn.Module):
    def __init__(
        self,
        num_layers,
        r1,
        r2,
        num_f_maps,
        num_classes,
        num_decoders,
    ):
        super(Predictor, self).__init__()
        self.decoders = nn.ModuleList(
            [
                copy.deepcopy(
                    Decoder(
                        num_layers,
                        r1,
                        r2,
                        num_f_maps,
                        num_classes,
                        num_classes,
                        att_type="sliding_att",
                        alpha=exponential_descrease(s),
                    )
                )
                for s in range(num_decoders)
            ]
        )  # num_decoders

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        """Forward pass."""
        mask = (x.sum(1).unsqueeze(1) != 0).int()
        out = self.conv_out(x) * mask[:, 0:1, :]
        outputs = out.unsqueeze(0)

        for decoder in self.decoders:
            out, x = decoder(
                F.softmax(out, dim=1) * mask[:, 0:1, :], x * mask[:, 0:1, :], mask
            )
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class ASFormer(Model):
    """
    An implementation of ASFormer
    """

    def __init__(
        self,
        num_decoders:int,
        num_layers:int,
        r1:float,
        r2:float,
        num_f_maps:int,
        input_dim:dict,
        num_classes:int,
        channel_masking_rate:float,
        state_dict_path:str=None,
        ssl_constructors:List=None,
        ssl_types:List=None,
        ssl_modules:List=None,
    ):
        input_dim = sum([x[0] for x in input_dim.values()])
        self.num_f_maps = int(float(num_f_maps))
        self.params = {
            "num_layers": int(float(num_layers)),
            "r1": float(r1),
            "r2": float(r2),
            "num_f_maps": self.num_f_maps,
            "input_dim": int(float(input_dim)),
            "channel_masking_rate": float(channel_masking_rate),
        }
        self.params_predictor = {
            "num_layers": int(float(num_layers)),
            "r1": r1,
            "r2": r2,
            "num_f_maps": self.num_f_maps,
            "num_classes": int(float(num_classes)),
            "num_decoders": int(float(num_decoders)),
        }
        super().__init__(ssl_constructors, ssl_modules, ssl_types, state_dict_path)

    def _feature_extractor(self) -> Union[torch.nn.Module, List]:
        return MyTransformer(**self.params)

    def _predictor(self) -> torch.nn.Module:
        return Predictor(**self.params_predictor)

    def features_shape(self) -> torch.Size:
        return torch.Size([self.num_f_maps])
