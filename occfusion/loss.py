import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner.amp import autocast

@autocast('cuda',torch.float32)
def geo_scal_loss(pred, ssc_target, semantic=True):

    # Get softmax probabilities
    if semantic:
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, 0, :, :, :]
    else:
        empty_probs = 1 - torch.sigmoid(pred)
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255 # 255 noise
    nonempty_target = ssc_target != 0 # 0 empty
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        (-torch.ones_like(precision)*torch.log(precision) - (1-torch.ones_like(precision))*torch.log(1-precision)) + \
        (-torch.ones_like(recall)*torch.log(recall) - (1-torch.ones_like(recall))*torch.log(1-recall)) + \
        (-torch.ones_like(spec)*torch.log(spec) - (1-torch.ones_like(spec))*torch.log(1-spec))
    )

@autocast('cuda',torch.float32)
def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = -torch.ones_like(precision)*torch.log(precision) - (1-torch.ones_like(precision))*torch.log(1-precision)
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = -torch.ones_like(recall)*torch.log(recall) - (1-torch.ones_like(recall))*torch.log(1-recall)
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = -torch.ones_like(specificity)*torch.log(specificity) - (1-torch.ones_like(specificity))*torch.log(1-specificity)
                loss_class += loss_specificity
            loss += loss_class
    return loss / count