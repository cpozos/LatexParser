import torch
from torch.functional import F

import math
from data.vocabulary import Vocabulary

def cal_loss(logits, targets):
    """
    args:
        logits: probability distribution return by model
                [B, MAX_LEN, voc_size]
        targets: target formulas
                [B, MAX_LEN]
    """

    # targets [B, MAX_LEN]
    padding = torch.ones_like(targets) * Vocabulary.PAD_TOKEN_ID # [1 1 1 1 ...]
    mask = (targets != padding)  # [False True True False]

    targets = targets.masked_select(mask)

    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, logits.size(2))
        ).contiguous().view(-1, logits.size(2))

    logits = torch.log(logits)

    assert logits.size(0) == targets.size(0)
    return F.nll_loss(logits, targets)

def cal_epsilon(k, step, method):
    """
    Reference:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
        See details in https://arxiv.org/pdf/1506.03099.pdf
    """
    assert method in ['inv_sigmoid', 'exp', 'teacher_forcing']

    if method == 'exp':
        return k**step
    elif method == 'inv_sigmoid':
        return k/(k+math.exp(step/k))
    else:
        return 1.