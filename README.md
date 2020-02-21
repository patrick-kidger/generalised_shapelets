Best Practices with Signatures. 
===============================
Evaluation of different signature methods on a variety of different datasets to evaluate most effective ways of utilising signatures. 


Setting up locally
------------------
To setup locally clone the repo, and create a conda environment and do a `pip install -r requirements.txt`. 

Download the following:
    • http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip
    • http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip
The password for the univariate set is `attempttoclassify`.

Stick the univariate data in `data/raw/univariate` and the multivariate in `data/raw/multivariate/`. This means we should have files in locations like `data/raw/univariate/Adiac/Adiac_TRAIN.arff`.

If locally, you may aswell remove some of the bigger files and keep ~10 or so univariate and multivariate folders. Now `run convert_data.py` which converts to pytorch format and saves in `data/interim`.

You will also need `raw/summaries`, the easiest way to get this is to download the corresponding folder from the AWS instance. After downloading and importing run `summarise.py` which makes summary dataframes and sends to `data/interim`.

Now you should be ready to go. Running is done through `main.py` and the hope is that the framework has sufficient flexibility to allow us to do most (all?) of the runs we are going to be interested in. Obviously we need to create a lot more functions and add more options to `main.py` but it should be straightforward. An example is pre-coded that looks at the effect of increasing order of the signature, the config is in `config.py` in root and the run is done in main.


The main.py file
-----------------
Runs can all be made through the main.py file (hopefully). Experiments should be setup through config.py where they have the following generic format:
```
config = {

    'basic_order': {    # Name of the experiment
           
        # Other options can go here

        # Grid of parameters to search over
        'param_grid': {
            # Data stuff
            'ds_name': ['Car', 'Chinatown', 'Computers'],   # Names of the datasets we wish to train on
            'multivariate': [False],    # Whether this is a multivariate or univariate run

            # Feature stuff
            'tfms': [   # Signature transforms, refer to src.features.add_features
                ['addtime'],
            ],
            'depth': [1, 2, 3, 4, 5, 6, 7, 8],  # Signature depth
            'logsig': [False],  # Whether to use the logsignature

            # Model stuff
            'cv_params': [  # Cross validation option. The great TS bakeoff used 100 resamples for every dataset.
                {
                    'n_splits': 3,  # Number of splits in each resample
                    'n_repeats': 10,    # Number of times to perform cv
                }
            ],
            'clf_key': ['lr'],  # Classifiers to use (see src.sklearn.parameter_grids)
            'metrics': [    # Metrics to get the CV scores of.
                {
                    'acc': make_scorer(accuracy_score)
                }
            ]
        }
    }

}
```

Runs can then be evaluated in `/notebooks/` using tools from `src.visualisation` and the `src.models.experiments.extractors` code.