# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np


def precision(tp, fp, fn):
    if tp + fp > 0:
        return tp / (tp + fp)
    return 0


def recall(tp, fp, fn):
    if tp + fn > 0:
        return tp / (tp + fn)
    return 0


def fscore(tp, fp, fn):
    p = precision(tp, fp, fn)
    r = recall(tp, fp, fn)
    if p + r > 0:
        return 2 * (p * r) / (p + r)
    return 0


def confusion_matrix(hat_y, y, n_classes=2):
    cnfm = np.zeros((n_classes, n_classes))
    for j in range(y.shape[0]):
        cnfm[y[j], hat_y[j]] += 1
    return cnfm


def scores_for_class(class_index, cnfm):
    tp = cnfm[class_index, class_index]
    fp = cnfm[:, class_index].sum() - tp
    fn = cnfm[class_index, :].sum() - tp
    tn = cnfm.sum() - tp - fp - fn

    p = precision(tp, fp, fn)
    r = recall(tp, fp, fn)
    f1 = fscore(tp, fp, fn)
    support = tp + tn
    return p, r, f1, support


def precision_recall_fscore_support(hat_y, y, n_classes=2):
    cnfm = confusion_matrix(hat_y, y, n_classes)

    if n_classes is None:
        n_classes = cnfm.shape[0]

    scores = np.zeros((n_classes, 4))
    for class_id in range(n_classes):
        scores[class_id] = scores_for_class(class_id, cnfm)
    return scores.T.tolist()


def make_loss_weights(nb_classes, target_idx, weight):
    """Creates a loss weight vector for nn.CrossEntropyLoss
    args:
        nb_classes: Number of classes
        target_idx: ID of the target (reweighted) class
        weight: Weight of the target class
    returns:
       weights (FloatTensor): Weight Tensor of shape `nb_classes` such that
                                  `weights[target_idx] = weight`
                                  `weights[other_idx] = 1.0`
    """

    weights = torch.ones(nb_classes)
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
        weights = weights.to(torch_device)
    weights[target_idx] = weight
    return weights
