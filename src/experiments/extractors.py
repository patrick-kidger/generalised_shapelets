"""
Methods for extracting data once an experiment has been run.
"""
from definitions import *


class ExperimentToFrame():
    """Converts a sacred run folder into a dataframe containing config and metrics information.

    Example:
        df = RunToFrame(ex_dir=MODELS_DIR + '/experiments/basic').generate_frame()
    """
    def __init__(self, ex_dir):
        """
        Args:
            ex_dir (str): The experiment run directory.
        """
        self.ex_dir = ex_dir

    def get_config(self, run_num):
        config = load_json(os.path.join(self.ex_dir, run_num, 'config.json'))

        del config['seed']

        if 'metrics' in config:
            config['metrics'] = config['metrics'].keys()

        return config

    def get_metrics(self, run_num):
        metrics = load_json(os.path.join(self.ex_dir, run_num, 'metrics.json'))

        # Strip of non-necessary entries
        metrics = {key: value['values'] for key, value in metrics.items()}

        return metrics

    def generate_frame(self):
        """ Main function for converting the info to a pd.DataFrame. """
        # List of run frames
        dataframes = []

        # Create a frame for each run
        for run_num in os.listdir(self.ex_dir):
            # Ignore _sources folder
            if run_num == '_sources':
                continue

            # Get config and metrics
            config = self.get_config(str(run_num))
            metrics = self.get_metrics(str(run_num))

            # Create a df from the information and add to dataframes
            df_config = pd.DataFrame.from_dict(config, orient='index').T
            df_metrics = pd.DataFrame.from_dict(metrics, orient='index').T
            df = pd.concat([df_config, df_metrics], axis=1)
            df.index = [int(run_num)]
            dataframes.append(df)

        # Concat for a full frame
        df = pd.concat(dataframes, axis=0, sort=True)
        df.sort_index(inplace=True)

        # Sort by run num
        df.sort_index(inplace=True)

        # Reorder some cols
        cols_front = ['ds_name']
        df = df[cols_front + [x for x in df.columns if x not in cols_front]]

        return df


if __name__ == '__main__':
    df = ExperimentToFrame(MODELS_DIR + '/experiments/basic_order').generate_frame()