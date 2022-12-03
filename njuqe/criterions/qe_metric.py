# Copyright (c) 2020 the NJUQE authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import numpy as np
from ..utils import precision_recall_fscore_support, make_loss_weights
from scipy.stats.stats import pearsonr


def precision_recall_fscore_support(y, y_hat, n_classes=2):
    confusion_matrix = np.zeros((n_classes, n_classes))
    for j in range(y.shape[0]):
        confusion_matrix[y[j], y_hat[j]] += 1

    scores = np.zeros((n_classes, 4))
    for class_id in range(n_classes):
        scores[class_id] = scores_for_class(class_id, confusion_matrix)
    return scores.T.tolist()

def scores_for_class(class_index, matrix):
    tp = matrix[class_index, class_index]
    fp = matrix[:, class_index].sum() - tp
    fn = matrix[class_index, :].sum() - tp
    tn = matrix.sum() - tp - fp - fn

    p, r, f1 = fscore(tp, fp, fn)
    support = tp + tn
    return p, r, f1, support

def precision(tp, fp):
    if tp + fp > 0:
        return tp / (tp + fp)
    return 0

def recall(tp, fn):
    if tp + fn > 0:
        return tp / (tp + fn)
    return 0

def fscore(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    if p + r > 0:
        return p, r, 2 * (p * r) / (p + r)
    return p, r, 0


def get_predictions_flat(target, predicted, pad_idx):
    mask = target != pad_idx
    token_indices = mask.view(-1).nonzero().squeeze()  # 获取所有不为tag_pad的index

    predicted_flat = predicted.view(-1, predicted.shape[-1]).squeeze()
    _, y_hat = predicted_flat.max(-1)

    y = target.view(-1)

    y = y[token_indices]
    y_hat = y_hat[token_indices]

    return y, y_hat


def compute_pearsonr(meters):
    y = meters["_score_labels"].y
    y_hat = meters["_score_labels"].y_hat
    if y is not None and len(y) > 2:
        pearson, _ = pearsonr(y_hat.detach().cpu().numpy(), y.detach().cpu().numpy())
    else:
        pearson = 0
    return round(pearson*100, 2)


def get_f1_compute_fn(name, type):
    def compute_f1(meters):
        y = meters["_"+name+"_labels"].y
        y_hat = meters["_"+name+"_labels"].y_hat
        if y is not None and len(y) > 2:
            _, _, f1, _ = precision_recall_fscore_support(y_hat, y)
            if type == "mult":
                return round(float(np.prod(f1))*100, 2)
            elif type == "ok":
                return round(float(f1[0]) * 100, 2)
            elif type == "bad":
                return round(float(f1[1]) * 100, 2)
        else:
            return 0.0
    return compute_f1
