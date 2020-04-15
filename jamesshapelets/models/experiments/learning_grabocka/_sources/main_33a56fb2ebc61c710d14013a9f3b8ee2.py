"""
main.py
=========================
Main experiment runner. The aim is for everything to be run through this file with different model configurations
imported via config
"""
from jamesshapelets.definitions import *
from sacred import Experiment

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy
from jamesshapelets.src.experiments.setup import create_fso, basic_gridsearch
from jamesshapelets.src.data.dicts import learning_ts_shapelets
from jamesshapelets.src.data.make_dataset import UcrDataset
from jamesshapelets.src.models.model import ShapeletNet
from jamesshapelets.src.models.dataset import PointsDataset, SigletDataset
from jamesshapelets.src.experiments.utils import ignite_accuracy_transform

import logging
logging.getLogger("ignite").setLevel(logging.WARNING)

import warnings
warnings.simplefilter('ignore', UserWarning)

# Experiment setup
ex_name = 'learning_ts_datasets'
ex = Experiment(ex_name)
save_dir = MODELS_DIR + '/experiments/{}'.format(ex_name)

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def run(args):
    # Open (needed for imap)
    ex_name, config = args

    # Setup experiment
    ex = Experiment(ex_name)

    import logging
    logger = logging.getLogger('logger')
    logger.setLevel(logging.CRITICAL)
    ex.logger = logger

    # Setup save directory, we append classifier if one exists
    save_dir = MODELS_DIR + '/experiments/{}'.format(ex_name)

    # Output verbosity
    verbose = 2
    if 'verbose' in config:
        verbose = config['verbose'][0]

    # Configuration, setup parameters that can vary here
    @ex.config
    def my_config():
        path_data = 'points'
        num_shapelets = 10
        window_size = 40
        aug_list = ['addtime']
        num_window_sizes = 5
        depth = 5
        discriminator = 'l2'
        max_epochs = 1000
        lr = 1e-2


    # Main run file
    @ex.main
    def main(_run, ds_name, path_tfm, num_shapelets, window_size, num_window_sizes, aug_list, depth, discriminator, max_epochs, lr):
        # Add in save_dir
        _run.save_dir = save_dir + '/' + _run._id

        # Get model training datsets
        ucr_train, ucr_test = UcrDataset(ds_name).get_original_train_test()
        if path_tfm == 'points':
            train_ds, test_ds = [PointsDataset(x.data, x.labels, window_size=window_size) for x in (ucr_train, ucr_test)]
        elif path_tfm == 'signature':
            train_ds, test_ds = [
                SigletDataset(x.data, x.labels, depth=depth, aug_list=aug_list, ds_length=x.size(1), num_window_sizes=num_window_sizes) for x in (ucr_train, ucr_test)
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
        model.to(device)
        loss_fn = nn.BCELoss() if ucr_train.n_classes == 2 else nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        # Setup
        trainer = create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)
        evaluator = create_supervised_evaluator(
            model=model,
            metrics={
                'acc': Accuracy(output_transform=ignite_accuracy_transform, is_multilabel=True if n_classes > 2 else False)
            },
            device=device
        )

        # Validation history
        validation_history = {
            'acc.train': [],
            'acc.test': [],
            'loss.train': [],
            'epoch': []
        }

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

                validation_history['acc.train'].append(train_acc)
                validation_history['acc.test'].append(test_acc)
                validation_history['loss.train'].append(trainer.state.output)
                validation_history['epoch'].append(epoch)

        # Time it
        start = time.time()
        trainer.run(train_dl, max_epochs=max_epochs)
        elapsed = time.time() - start

        _run.log_scalar(elapsed, 'training_time')
        _run.log_scalar(validation_history['acc.train'][-1], 'acc.train')
        _run.log_scalar(validation_history['acc.test'][-1], 'acc.test')
        _run.log_scalar(validation_history['loss.train'][-1], 'loss.train')

        save_pickle(validation_history, save_dir + '/validation_history.pkl')

    # Create FSO (this creates a folder to log information into).
    create_fso(ex, save_dir, remove_folder=False)

    # Run a gridsearch over all parameter combinations.
    basic_gridsearch(ex, config)


if __name__ == '__main__':
    learning_ts_shapelets = [
        'Coffee'
    ]
    param_grid = {
        # 'ds_name': learning_ts_shapelets[0:10],
        'ds_name': ['Beef'],
        'path_tfm': ['signature'],

        'num_shapelets': [50],
        'num_window_sizes': [5],
        'max_window': [1000],
        'depth': [5],

        'discriminator': ['linear'],

        'max_epochs': [100],
        'lr': [1e-1],
    }

    run(('test', param_grid))

