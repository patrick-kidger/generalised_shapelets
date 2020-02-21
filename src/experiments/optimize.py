"""
optimize.py
===========================
Hyper-parameter optimization using tune.
"""
from definitions import *
import numpy as np
import ray
from ray import tune
from ray.tune import Trainable
from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.models.grids import CLASSIFIER_PARAM_GRIDS


class ModelTrainer(Trainable):
    """ Trainable class for hyperparameter searching with ray.tune. """
    def _setup(self, config):
        # Get model builder and data
        self.model_builder, self.dataset = config['args']
        del config['args']

        # Setup model
        self.model = self.model_builder.set_hyperparameters(config).build_model()
        print(config)

    def _train(self):
        # Train
        X, y = self.dataset.to_ml()
        cv = StratifiedKFold(n_splits=5).split(X, y)
        print('this the hard bit')
        print(X.shape)
        acc = cross_val_score(self.model, X, y.view(-1), cv=cv, n_jobs=1)
        # acc=2
        print('finite')

        # Metrics
        metrics = {
            'mean_accuracy': np.mean(acc)
        }

        return metrics

    def _save(self, checkpoint_dir):
        return None

    def _restore(self, checkpoint_path):
        return None


def convert_and_save_analysis(_run, analysis):
    """ Updates the run configuration with the best found configuration. """
    # Save results from each configuration
    results = analysis.dataframe()
    results = results[['mean_accuracy'] + [x for x in results.columns if ('config/' in x) and ('/args' not in x)]]
    save_pickle(results, _run.save_dir + '/results.pkl')

    # Get the best config (remove the args we inserted)
    best_config = analysis.get_best_config(metric="mean_accuracy")
    del best_config['args']

    # Save into the original configuration
    file = _run.save_dir + '/config.json'

    with open(file, 'r') as jsonFile:
        config = json.load(jsonFile)

    for b_key, b_val in best_config.items():
        config[b_key] = b_val

    with open(file, 'w') as jsonFile:
        json.dump(config, jsonFile)

    return best_config


@timeit
def optimize_hyperparameters(_run, model_builder, dataset, clf):
    # Local mode for debug
    # ray.init(num_cpus=16, num_gpus=0)
    ray.init(num_cpus=4)

    # Get the space to iterate over
    space = CLASSIFIER_PARAM_GRIDS[clf]

    # Add args for use inside _train
    space['args'] = model_builder, dataset

    # HyperOpt search
    algo = HyperOptSearch(
        space=space,
        max_concurrent=16,
        metric='mean_accuracy',
        mode='max'
    )

    analysis = tune.run(
        ModelTrainer,
        search_alg=algo,
        stop={
            "training_iteration": 1,
        },
        num_samples=30,
        checkpoint_freq=0,
        verbose=1,
        sync_on_checkpoint=False,
        # resources_per_trial={'cpu': 1, 'gpu': 0}
    )

    # Get best config and save some info
    best_config = convert_and_save_analysis(_run, analysis)

    # Close ray
    ray.shutdown()

    return best_config




