from typing import Dict, List

import torch.nn.functional as F
import torch.nn as nn
import torch

class SV_regularization(nn.Module):

    def __init__(self):
        super(SV_regularization, self).__init__()

    def forward(self, linear_weights: torch.tensor) -> torch.float:

        _, s, v = torch.svd(torch.matmul(linear_weights, linear_weights.T))

        ratio = s[0]/(s[-1] + 0.00001)
        entropy = torch.sum(F.softmax(torch.sqrt(s), dim=0) * F.log_softmax(torch.sqrt(s), dim=0))
        norm = torch.mean(torch.norm(linear_weights, dim=1))

        return ratio, entropy, norm