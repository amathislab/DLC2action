#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
import torch
from torch import nn
from torch.nn import functional as F
import math
import copy
from dlc2action.model.base_model import Model


class _Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(_Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class _FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class _MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = _attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat).transpose(1, 2)

        return output


def _attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.BatchNorm1d(d_model)
        self.norm_2 = nn.BatchNorm1d(d_model)
        self.attn = _MultiHeadAttention(heads, d_model)
        self.ff = _FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class _Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.pe = _PositionalEncoder(d_model)
        self.layers = _get_clones(_EncoderLayer(d_model, heads), N)
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, src, mask):
        x = self.pe(src)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class _PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
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
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(-1)
        x = x + self.pe[:, :, :seq_len].to(x.device)
        return x


class _TransformerModule(nn.Module):
    def __init__(
        self, heads, N, d_model, input_dim, output_dim, num_pool=3, add_batchnorm=True
    ):
        super(_TransformerModule, self).__init__()
        self.encoder = _Encoder(d_model, N, heads)
        self.in_layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        layer = nn.ModuleList()
        layer.append(nn.Conv1d(input_dim, d_model, 3, padding=1))
        layer.append(nn.ReLU())
        if num_pool > 0:
            layer.append(nn.MaxPool1d(2, 2))
        self.in_layers.append(layer)
        for _ in range(num_pool - 1):
            layer = nn.ModuleList()
            layer.append(nn.Conv1d(d_model, d_model, 3, padding=1))
            layer.append(nn.ReLU())
            if add_batchnorm:
                layer.append(nn.BatchNorm1d(d_model))
            layer.append(nn.MaxPool1d(2, 2))
            self.in_layers.append(layer)
        for _ in range(num_pool):
            layer = nn.ModuleList()
            layer.append(nn.Conv1d(d_model, d_model, 3, padding=1))
            layer.append(nn.ReLU())
            if add_batchnorm:
                layer.append(nn.BatchNorm1d(d_model))
            self.out_layers.append(layer)
        self.conv_out = nn.Conv1d(d_model, output_dim, 3, padding=1)

    def forward(self, x):
        sizes = []
        for layer_list in self.in_layers:
            sizes.append(x.shape[-1])
            for layer in layer_list:
                x = layer(x)
        mask = (x.sum(1).unsqueeze(1) != 0).int()
        x = self.encoder(x, mask)
        sizes = sizes[::-1]
        for i, (layer_list, size) in enumerate(zip(self.out_layers, sizes)):
            for layer in layer_list:
                x = layer(x)
            x = F.interpolate(x, size)
        x = self.conv_out(x)
        return x


class _Predictor(nn.Module):
    def __init__(self, dim, num_classes):
        super(_Predictor, self).__init__()
        self.conv_out_1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv_out_2 = nn.Conv1d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv_out_1(x)
        x = F.relu(x)
        x = self.conv_out_2(x)
        return x


class Transformer(Model):
    """
    A modification of Transformer-Encoder with additional max-pooling and upsampling

    Set `num_pool` to 0 to get a standart transformer-encoder.
    """

    def __init__(
        self,
        N,
        heads,
        num_f_maps,
        input_dim,
        num_classes,
        num_pool,
        add_batchnorm=False,
        feature_dim=None,
        state_dict_path=None,
        ssl_constructors=None,
        ssl_types=None,
        ssl_modules=None,
    ):
        input_dim = sum([x[0] for x in input_dim.values()])
        if feature_dim is None:
            feature_dim = num_classes
            self.f_shape = None
            self.params_predictor = None
        else:
            self.f_shape = torch.Size([int(feature_dim)])
            self.params_predictor = {
                "dim": int(feature_dim),
                "num_classes": int(num_classes),
            }
        self.params = {
            "d_model": int(num_f_maps),
            "input_dim": int(input_dim),
            "N": int(N),
            "heads": int(heads),
            "add_batchnorm": add_batchnorm,
            "num_pool": int(num_pool),
            "output_dim": int(feature_dim),
        }
        super().__init__(ssl_constructors, ssl_modules, ssl_types, state_dict_path)

    def _feature_extractor(self):
        return _TransformerModule(**self.params)

    def _predictor(self) -> torch.nn.Module:
        if self.params_predictor is None:
            return nn.Identity()
        else:
            return _Predictor(**self.params_predictor)

    def features_shape(self) -> torch.Size:
        return self.f_shape
