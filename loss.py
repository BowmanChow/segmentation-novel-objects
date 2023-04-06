import torch
from torch import nn


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        print(
            f"{type(self).__name__} init | reduction : {self.reduction} | alpha : {self.alpha} "
        )

    def forward(self, input_proba, target_proba, mask=None):

        outputs = torch.log(input_proba)
        labels = target_proba

        loss = (outputs * labels).mean(dim=1)

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs
