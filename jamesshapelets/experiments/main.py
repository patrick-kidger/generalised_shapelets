"""
main.py
=========================
Main experiment runner. The aim is for everything to be run through this file with different model configurations
imported via config
"""
from definitions import *
from sacred import Experiment

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy
from sklearn.metrics import accuracy_score, roc_auc_score
from src.experiments.setup import create_fso, basic_gridsearch
from src.data.make_dataset import UcrDataset
from src.models.model import ShapeletNet
from src.models.dataset import PointsDataset, SigletDataset
from src.experiments.utils import ignite_accuracy_transform

import logging
logging.getLogger("ignite").setLevel(logging.WARNING)

import warnings
warnings.simplefilter('ignore', UserWarning)

# Experiment setup
ex_name = 'test'
ex = Experiment(ex_name)
save_dir = MODELS_DIR + '/experiments/{}'.format(ex_name)


# Configuration, setup parameters that can vary here
@ex.config
def my_config():
    path_data = 'points'
    num_shapelets = 10
    window_size = 40
    discriminator = 'l2'
    max_epochs = 1000
    lr = 1e-2


# Main run file
@ex.main
def main(_run, ds_name, path_tfm, num_shapelets, window_size, discriminator, max_epochs, lr):
    # Add in save_dir
    _run.save_dir = save_dir + '/' + _run._id

    # Get model training datsets
    ucr_train, ucr_test = UcrDataset(ds_name).get_original_train_test()
    if path_tfm == 'points':
        train_ds, test_ds = [PointsDataset(x.data, x.labels, window_size=window_size) for x in (ucr_train, ucr_test)]
    elif path_tfm == 'signature':
        train_ds, test_ds = [
            SigletDataset(x.data, x.labels, window_size=window_size, depth=5, aug_list=['addtime'], ds_length=x.size(1), num_window_size=5) for x in (ucr_train, ucr_test)
        ]
    n_classes = ucr_train.n_classes
    n_outputs = n_classes - 1 if n_classes == 2 else n_classes

    # Loaders
    train_dl = DataLoader(train_ds, batch_size=32)
    test_dl = DataLoader(test_ds, batch_size=test_ds.size(0))

    # Setup
    model = ShapeletNet(
        num_shapelets=num_shapelets,
        shapelet_len=train_ds.shapelet_len,
        num_outputs=n_outputs,
        init_data=train_ds.data,
        discriminator=discriminator
    )
    loss_fn = nn.BCELoss() if ucr_train.n_classes == 2 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    # Setup
    trainer = create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)
    evaluator = create_supervised_evaluator(
        model=model,
        metrics={
            'acc': Accuracy(output_transform=ignite_accuracy_transform, is_multilabel=True if n_classes > 2 else False)
        }
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(trainer):
        epoch = trainer.state.epoch
        if epoch % 50 == 0:
            evaluator.run(train_dl)
            train_acc = evaluator.state.metrics['acc']
            evaluator.run(test_dl)
            test_acc = evaluator.state.metrics['acc']
            print('EPOCH: [{}]'.format(epoch))
            print('Train loss: {} - Train acc: {:.2f}%'.format(trainer.state.output, 100 * train_acc))
            print('Acc Test: {:.2f}%'.format(100 * test_acc))

    trainer.run(train_dl, max_epochs=max_epochs)


if __name__ == '__main__':
    param_grid = {
        'ds_name': ['Coffee'],
        'path_tfm': ['signature'],
        'num_shapelets': [100],
        'window_size': [40],
        'discriminator': ['linear'],
        'max_epochs': [1000],
        'lr': [1e-1],
    }

    # Create FSO (this creates a folder to log information into).
    create_fso(ex, save_dir)

    # Run a gridsearch over all parameter combinations.
    basic_gridsearch(ex, param_grid, tqdm_progress=True)
