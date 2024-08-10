import os
import numpy as np

import os
import torch.nn.functional as F
dir_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(dir_path)
import torch.nn as nn
import torch

class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        batch_size = f_s.size(0)
        loss = 0
        for i in range(batch_size):
            student = f_s[i].view(f_s[i].shape[0], -1)
            teacher = f_t[i].view(f_t[i].shape[0], -1)

            # RKD distance loss
            with torch.no_grad():
                t_d = self.pdist(teacher, squared=False)
                mean_td = t_d[t_d > 0].mean()
                t_d = t_d / mean_td

            d = self.pdist(student, squared=False)
            mean_d = d[d > 0].mean()
            d = d / mean_d

            loss_d = F.smooth_l1_loss(d, t_d)

            # RKD Angle loss
            with torch.no_grad():
                td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
                norm_td = F.normalize(td, p=2, dim=2)
                t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

            sd = (student.unsqueeze(0) - student.unsqueeze(1))
            norm_sd = F.normalize(sd, p=2, dim=2)
            s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

            loss_a = F.smooth_l1_loss(s_angle, t_angle)

            loss += self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, device):
        super(TripletLoss, self).__init__()
        self.device = device
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features, text_features):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(image_features, text_features.transpose(0, 1))
        logits1 = logit_scale * torch.matmul(text_features, image_features.transpose(0, 1))
        labels = torch.tensor(np.arange(12)).to(self.device)
        loss1 = F.cross_entropy(logits, labels)
        loss2 = F.cross_entropy(logits1, labels)
        loss = loss1 + loss2
        return loss

class SD_Constration(nn.Module):
    def __init__(self, hidden_dim):
        super(SD_Constration, self).__init__()
        ##########################################
        # private encoders
        ##########################################
        self.private_c = nn.Sequential()
        self.private_c.add_module('private_t_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.private_c.add_module('private_t_activation_1', nn.Sigmoid())

        self.private_p = nn.Sequential()
        self.private_p.add_module('private_v_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.private_p.add_module('private_v_activation_1', nn.Sigmoid())

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())


    def forward(self, current_image, prior_image):
        # Private-shared components
        utt_private_c = self.private_c(current_image.squeeze(1))
        utt_private_p = self.private_p(prior_image.squeeze(1))

        utt_shared_c = self.shared(current_image.squeeze(1))
        utt_shared_p = self.shared(prior_image.squeeze(1))

        return utt_shared_c, utt_shared_p, utt_private_c, utt_private_p
