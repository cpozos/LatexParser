import torch
from torch.functional import F

from data.vocabulary import Vocabulary

def collate_fn(token2id, batch):
    # filter the pictures that have different weight or height
    size = batch[0][0].size()
    batch = [img_formula for img_formula in batch if img_formula[0].size() == size]

    # sort by the length of formula
    batch.sort(key=lambda img_formula: len(img_formula[1].split()), reverse=True)
    imgs, formulas = zip(*batch)
    formulas = [formula.split() for formula in formulas]

    # targets for training , begin with START_TOKEN
    tgt4training = formulas2tensor(add_start_token(formulas), token2id)

    # targets for calculating loss , end with END_TOKEN
    tgt4cal_loss = formulas2tensor(add_end_token(formulas), token2id)
    imgs = torch.stack(imgs, dim=0)

    return imgs, tgt4training, tgt4cal_loss

def formulas2tensor(formulas, token2id):
    """
        Makes the convertion from latex formula to tensor
    """
    batch_size = len(formulas)
    max_len = len(formulas[0])
    tensors = torch.ones(batch_size, max_len, dtype=torch.long) * Vocabulary.PAD_TOKEN_ID
    for i, formula in enumerate(formulas):
        for j, token in enumerate(formula):
            tensors[i][j] = token2id.get(token, Vocabulary.UNK_TOKEN_ID)
    return tensors

def add_start_token(formulas):
    return [['<s>']+formula for formula in formulas]

def add_end_token(formulas):
    return [formula+['</s>'] for formula in formulas]

def cal_loss(logits, targets):
    """
    args:
        logits: probability distribution return by model
                [B, MAX_LEN, voc_size]
        targets: target formulas
                [B, MAX_LEN]
    """

    # targets [1 29 34 1]
    padding = torch.ones_like(targets) * Vocabulary.PAD_TOKEN_ID # [1 1 1 1]
    mask = (targets != padding)  # [False True True False]

    targets = targets.masked_select(mask)

    mask_2 = mask.unsqueeze(2).expand(-1, -1, logits.size(2))
    logits = logits.masked_select(mask_2).contiguous().view(-1, logits.size(2))
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