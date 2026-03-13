import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from typing import Tuple
from pointnet.models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
import copy


class PointNetClassifier(nn.Module):
    def __init__(self, n_classes, normal_channel=True, add_fc=True):
        super(PointNetClassifier, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        # set n_classes to None to disable the fc layer
        # self.fc3 = nn.Linear(256, n_classes)
        self.fc3 = None if add_fc else nn.Linear(256, n_classes)
        
        # ---------------------------------
        # helper layers
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat


    def update_fc(self, n_classes):
        self.fc3 = nn.Linear(256, n_classes)
        # self.to(self.device)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class PointNetLoss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(PointNetLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor, trans_feat: torch.Tensor) -> torch.Tensor:
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

