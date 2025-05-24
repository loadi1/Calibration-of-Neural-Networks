import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DualFocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(DualFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logp_k = F.log_softmax(input, dim=-1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        loss = -1 * (1 - p_k + p_j) ** self.gamma * logp_k

        if self.size_average: return loss.mean()
        else: return loss.sum()


class DualFocalLossGra(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(DualFocalLossGra, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logp_k = F.log_softmax(input, dim=-1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k


        with torch.no_grad():
            p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()
        f_p = (1 - p_k + p_j) ** self.gamma
        loss = -1 * f_p * logp_k

        if self.size_average: return loss.mean()
        else: return loss.sum()        


class DualFocalLossExp(nn.Module):
    def __init__(self, gamma=0, size_average=False, temperature=1.0):
        super(DualFocalLossExp, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.temperature = temperature

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logp_k = F.log_softmax(input, dim=-1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        loss = -1 * (1 - p_k + p_j) ** self.gamma * logp_k

        with torch.no_grad():
            weight = torch.exp(
                torch.clamp(loss.detach(), min=0, max=self.temperature) / (self.temperature + 1)
            ) - 1
        loss = loss * weight

        if self.size_average: return loss.mean()
        else: return loss.sum()