import collections as co
import copy
import math
import torch
import tqdm
import torchshapelets


def normalise_data(X, eps=1e-5):
    # X is assumed to be of shape (..., length, channel)
    out = []
    for Xi in X.unbind(dim=-1):
        mean = Xi.mean()
        std = Xi.std()
        out.append((Xi - mean) / (eps + std))
    return torch.stack(out, dim=-1)


def dataloader(dataset, **kwargs):
    if 'shuffle' not in kwargs:
        kwargs['shuffle'] = True
    if 'drop_last' not in kwargs:
        kwargs['drop_last'] = True
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = 32
    kwargs['batch_size'] = min(kwargs['batch_size'], len(dataset))
    return torch.utils.data.DataLoader(dataset, **kwargs)


def get_sample_batch(dataloader, num_shapelets_per_class, num_shapelets):
    batch_elems = []
    y_seen = co.defaultdict(int)
    while True:  # in case we need to iterate through the same dataloader multiple times to find the same samples again
        for X, y in dataloader:
            for Xi, yi in zip(X, y):
                yi = int(yi)
                if y_seen[yi] < num_shapelets_per_class:
                    batch_elems.append(Xi)
                    y_seen[yi] += 1
                if len(batch_elems) == num_shapelets:
                    return torch.stack(batch_elems, dim=0)
        # len(y_seen) should now be the number of classes
        if len(y_seen) * num_shapelets_per_class != num_shapelets:
            raise RuntimeError("Could not get a sample batch: Have been told that there should {} shapelets per class, "
                               "and {} shaplets in total, but only found {} classes.".format(num_shapelets_per_class,
                                                                                             num_shapelets,
                                                                                             len(y_seen)))


def _count_parameters(model):
    """Counts the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad_)


def _compute_binary_accuracy(pred_y, true_y):
    """Computes the accuracy of a classifier.

    Arguments:
        pred_y: A one dimensional tensor of floats (a sigmoid will be applied to determine the classification result).
        true_y: A one dimensional tensor of floats; 1.0 corresponds to one class, 0.0 corresponds to the other.

    Returns:
        A single float describing the accuracy.
    """
    assert pred_y.shape == true_y.shape, "Shape mismatch: pred_y has shape {}, true_y has shape {}".format(pred_y.shape,
                                                                                                           true_y.shape)

    binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(true_y.dtype)
    prediction_matches = (binary_prediction == true_y).to(true_y.dtype)
    proportion_correct = prediction_matches.sum() / true_y.size(0)
    return proportion_correct


def _compute_multiclass_accuracy(pred_y, true_y):
    """Computes the accuracy of a classifier.

    Arguments:
        pred_y: A two dimensional tensor of floats, of shape (batch, targets).
        true_y: A one dimensional tensor of targets; integers corresponding to each class.

    Returns:
        A single float describing the accuracy.
    """
    prediction = torch.argmax(pred_y, dim=1)
    prediction_matches = (prediction == true_y).to(pred_y.dtype)
    proportion_correct = prediction_matches.sum() / true_y.size(0)
    return proportion_correct


class _AttrDict(dict):
    def __getattr__(self, item):
        return self[item]


def _evaluate_metrics(dataloader, model, times, loss_fn, num_classes):
    with torch.no_grad():
        accuracy_fn = _compute_binary_accuracy if num_classes == 2 else _compute_multiclass_accuracy
        total_loss = 0
        total_accuracy = 0
        total_dataset_size = 0
        for batch in dataloader:
            X, y = batch
            batch_size = y.size(0)
            pred_y, _, _, _ = model(times, X)
            total_accuracy += accuracy_fn(pred_y, y) * batch_size
            total_loss += loss_fn(pred_y, y) * batch_size
            total_dataset_size += batch_size
        total_loss /= total_dataset_size  # assume 'mean' reduction in the loss function
        total_accuracy /= total_dataset_size
        return _AttrDict(loss=total_loss, accuracy=total_accuracy)


def train_loop(train_dataloader, val_dataloader, model, times, optimizer, loss_fn, epochs, num_classes):
    """Standard training loop.

    Has a few simple bells and whistles:
    - Decreases learning rate on plateau.
    - Stops training if there's no improvement in training loss for several epochs.
    - Uses the best model (measured by validation accuracy) encountered during training, not just the final one.
    """
    model.train()
    best_model = model
    best_train_loss = math.inf
    best_val_accuracy = 0
    best_epoch = 0
    history = []
    breaking = False

    epoch_per_metric = 10
    plateau_patience = 1  # this will be multiplied by epoch_per_metric for the actual patience
    plateau_terminate = 50
    similarity_coefficient = 0.1
    length_coefficient = 0.1
    pseudometric_coefficient = 0.1

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=plateau_patience, mode='max')

    tqdm_range = tqdm.tqdm(range(epochs))
    try:
        for epoch in tqdm_range:
            if breaking:
                break
            for batch in train_dataloader:
                if breaking:
                    break
                X, y = batch
                pred_y, shapelet_similarity, shapelet_lengths, discrepancy_fn = model(times, X)
                loss = loss_fn(pred_y, y)
                loss = loss + similarity_coefficient * torchshapelets.similarity_regularisation(shapelet_similarity)
                loss = loss + length_coefficient * torchshapelets.length_regularisation(shapelet_lengths)
                loss = loss + pseudometric_coefficient * torchshapelets.pseudometric_regularisation(discrepancy_fn)
                loss.backward()
                optimizer.step()
                model.clip_length()
                optimizer.zero_grad()

            if epoch % epoch_per_metric == 0 or epoch == epochs - 1:
                model.eval()
                train_metrics = _evaluate_metrics(train_dataloader, model, times, loss_fn, num_classes)
                val_metrics = _evaluate_metrics(val_dataloader, model, times, loss_fn, num_classes)
                model.train()

                if train_metrics.loss * 1.0001 < best_train_loss:
                    best_train_loss = train_metrics.loss
                    best_epoch = epoch

                if val_metrics.accuracy > best_val_accuracy:
                    best_val_accuracy = val_metrics.accuracy
                    del best_model  # so that we don't have three copies of a model simultaneously
                    best_model = copy.deepcopy(model)

                tqdm_range.write('Epoch: {}  Train loss: {:.3}  Train accuracy: {:.3}  Val loss: {:.3}  '
                                 'Val accuracy: {:.3}'
                                 ''.format(epoch, train_metrics.loss, train_metrics.accuracy, val_metrics.loss,
                                           val_metrics.accuracy))
                scheduler.step(val_metrics.accuracy)
                history.append(_AttrDict(epoch=epoch, train_loss=train_metrics.loss,
                                         train_accuracy=train_metrics.accuracy,
                                         val_loss=val_metrics.loss, val_accuracy=val_metrics.accuracy))

                if epoch > best_epoch + plateau_terminate:
                    tqdm_range.write('Breaking because of no improvement in training loss for {} epochs.'
                                     ''.format(plateau_terminate))
                    breaking = True
    except KeyboardInterrupt:
        tqdm_range.write('Breaking because of keyboard interrupt.')

    for parameter, best_parameter in zip(model.parameters(), best_model.parameters()):
        parameter.data = best_parameter.data
    return history


def evaluate_model(train_dataloader, val_dataloader, test_dataloader, model, times, loss_fn, history, num_classes):
    model.eval()
    train_metrics = _evaluate_metrics(train_dataloader, model, times, loss_fn, num_classes)
    val_metrics = _evaluate_metrics(val_dataloader, model, times, loss_fn, num_classes)
    test_metrics = _evaluate_metrics(test_dataloader, model, times, loss_fn, num_classes)

    return _AttrDict(times=times,
                     num_classes=num_classes,
                     train_dataloader=train_dataloader,
                     val_dataloader=val_dataloader,
                     test_dataloader=test_dataloader,
                     model=model,
                     parameters=_count_parameters(model),
                     history=history,
                     train_metrics=train_metrics,
                     val_metrics=val_metrics,
                     test_metrics=test_metrics)


class LinearShapeletTransform(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_shapelets, num_shapelet_samples, discrepancy_fn,
                 max_shapelet_length, num_continuous_samples):
        super(LinearShapeletTransform, self).__init__()

        self.shapelet_transform = torchshapelets.GeneralisedShapeletTransform(in_channels=in_channels,
                                                                              num_shapelets=num_shapelets,
                                                                              num_shapelet_samples=num_shapelet_samples,
                                                                              discrepancy_fn=discrepancy_fn,
                                                                              max_shapelet_length=max_shapelet_length,
                                                                              num_continuous_samples=num_continuous_samples)
        self.linear = torch.nn.Linear(num_shapelets, out_channels)

    def forward(self, times, X):
        shapelet_similarity = self.shapelet_transform(times, X)
        out = self.linear(shapelet_similarity.log())
        if out.size(-1) == 1:
            out.squeeze(-1)
        return out, shapelet_similarity, self.shapelet_transform.lengths, self.shapelet_transform.discrepancy_fn

    def clip_length(self):
        self.shapelet_transform.clip_length()

    def set_shapelets(self, times, path):
        self.shapelet_transform.reset_parameters(times, path)
