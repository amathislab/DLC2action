#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
# Incorporates code adapted from MotionBERT by Walter0807
# Original work Copyright (c) 2023 Walter0807
# Source: https://github.com/Walter0807/MotionBERT
# Originally licensed under Apache License Version 2.0, 2023
# Combined work licensed under GNU AGPLv3
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from dlc2action.model.base_model import Model
from einops import rearrange
from dlc2action.model.motionbert_modules import DSTformer
from functools import partial


def load_backbone(args):
    model_backbone = DSTformer(dim_in=args['dim'], dim_out=args["dim"], dim_feat=args["dim_feat"], dim_rep=args["dim_rep"],
                                   depth=args["depth"], num_heads=args["num_heads"], mlp_ratio=args["mlp_ratio"], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   maxlen=args["maxlen"], num_joints=args["num_joints"])
    return model_backbone


class ActionHeadClassification(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=60, num_joints=17, hidden_dim=2048):
        super(ActionHeadClassification, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat = self.fc2(feat)
        return feat

class ActionHeadSegmentation(nn.Module):
    def __init__(self, input_dim, dropout_ratio=0., num_classes=60, hidden_dim=128):
        super(ActionHeadSegmentation, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat):
        '''
            Input: (N, M, T, F)
        '''
        # print('FEAT before', feat.shape)
        feat = self.dropout(feat)
        feat = self.fc1(feat)
        feat = rearrange(feat, "n t f -> n f t")
        feat = self.bn(feat)
        feat = rearrange(feat, "n f t -> n t f")
        feat = self.relu(feat)
        feat = self.fc2(feat)
        out = rearrange(feat, "n t f -> n f t")
        # print('FEAT after -out', feat.shape)
        return out

class ActionHeadEmbed(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_joints=17, hidden_dim=2048):
        super(ActionHeadEmbed, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = F.normalize(feat, dim=-1)
        return feat

class ActionNet(nn.Module):
    def __init__(self, backbone, channels):
        super(ActionNet, self).__init__()
        self.backbone = backbone
        self.channels = channels
        # if version=='class':
        #     self.head = ActionHeadClassification(dropout_ratio=dropout_ratio, dim_rep=dim_rep, num_classes=num_classes, num_joints=num_joints)
        # elif version=='embed':
        #     self.head = ActionHeadEmbed(dropout_ratio=dropout_ratio, dim_rep=dim_rep, hidden_dim=hidden_dim, num_joints=num_joints)
        # else:
        #     raise Exception('Version Error.')

    def forward(self, x):
        '''
            Input: (N, M x T x J x 3)
        '''
        #print('BEFORE', x.shape)
        x = rearrange(x, 'n (j c) t -> n t j c', c=self.channels)
        # print('AFTER', x.shape)
        # N, T, J, C = x.shape
        feat = self.backbone.get_representation(x)
        # feat = feat.reshape([N, 1, T, self.feat_J, -1])      # (N, M, T, J, C)
        # out = self.head(feat)
        feat = rearrange(feat, "n t j c -> n t (j c)")
        return feat


class MotionBERT(Model):
    """
    An implementation of MotionBERT
    """

    def __init__(
        self,
        dim_feat,
        dim_rep,
        depth,
        num_heads,
        mlp_ratio,
        len_segment,
        num_joints,
        num_classes,
        input_dims,
        state_dict_path=None,
        ssl_constructors=None,
        ssl_types=None,
        ssl_modules=None,
    ):
        if dim_rep == "dim_feat":
            dim_rep = dim_feat
        input_dims = int(sum([s[0] for s in input_dims.values()]))
        print('input_dims', input_dims)
        print('num_joints', num_joints)
        assert input_dims % num_joints == 0
        args = {
            "dim_feat": int(dim_feat),
            "dim_rep": int(dim_rep),
            "depth": int(depth),
            "num_heads": int(num_heads),
            "mlp_ratio": int(mlp_ratio),
            "maxlen": int(len_segment),
            "num_joints": int(num_joints),
            "dim": int(input_dims // num_joints),
        }
        self.f_shape = args["dim_rep"] * args["num_joints"]
        self.params = {
            "backbone": load_backbone(args),
            "channels": int(input_dims // num_joints),
        }
        self.num_classes = num_classes
        super().__init__(ssl_constructors, ssl_modules, ssl_types, state_dict_path)

    def _feature_extractor(self):
        return ActionNet(**self.params)

    def _predictor(self) -> torch.nn.Module:
        #return ActionHeadSegmentation(dropout_ratio=0.5, input_dim=self.f_shape, num_classes=4)
        return ActionHeadSegmentation(dropout_ratio=0.1, input_dim=self.f_shape, num_classes=self.num_classes)

    def features_shape(self) -> torch.Size:
        return self.f_shape
