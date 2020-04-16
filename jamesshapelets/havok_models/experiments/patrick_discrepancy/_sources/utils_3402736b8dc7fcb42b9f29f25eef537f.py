import torch
from torch.nn.functional import one_hot


def ignite_accuracy_transform(output):
    """ Transform necessary for computing the accuracy with ignite. """
    preds, labels = output
    if preds.shape[1] == 1:
        return [(preds > 0.5).int(), labels]
    else:
        labels_ohe = one_hot(labels, preds.size(1))
        preds_ohe = one_hot(torch.argmax(torch.nn.Sigmoid()(preds), dim=1), preds.size(1))
        return [preds_ohe, labels_ohe]
