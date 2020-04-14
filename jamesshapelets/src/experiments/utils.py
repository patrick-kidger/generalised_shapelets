import torch
from torch.nn.functional import one_hot


def ignite_accuracy_transform(output):
    """ Transform necessary for computing the accuracy with ignite. """
    preds, labels = output
    if preds.shape[1] == 1:
        return [(preds > 0.5).int(), labels]
    else:
        labels_ohe = one_hot(labels, preds.size(1))
        preds = one_hot(preds.argmax(dim=1), preds.size(1))
        return [preds, labels_ohe]
