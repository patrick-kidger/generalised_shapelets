"""
Gets summary information for the datasets and converts to a use-able format.
"""
from definitions import *
import pandas as pd
import numpy as np


def clean_uv(df):
    df.rename(columns={'Test ': 'Test', 'Train ': 'Train'}, inplace=True)
    df.drop(['ï»¿ID', 'Data donor/editor'], axis=1, inplace=True)
    return df


def clean_mv(df):
    """ For loading and cleaning the multivariate summary. """
    # Change class counts to be a list rather than loads of columns
    cols = ['ClassCounts'] + [x for x in df.columns if 'Unnamed:' in x]
    df['ClassCounts'] = df[cols].apply(lambda x: [int(x) for x in list(x) if ~np.isnan(x)], axis=1)
    df.drop(cols[1:], axis=1, inplace=True)
    return df


if __name__ == '__main__':
    # Load full summaries
    uv = pd.read_csv(DATA_DIR + '/raw/summaries/univariate.csv', encoding='latin-1', index_col=2)
    mv = pd.read_csv(DATA_DIR + '/raw/summaries/multivariate.csv', index_col=0)

    # Clean
    mv = clean_mv(mv)
    uv = clean_uv(uv)

    # Add type col to mv
    mv['Type'] = np.nan

    # Save
    save_pickle(uv, DATA_DIR + '/interim/summaries/univariate.pkl')
    save_pickle(mv, DATA_DIR + '/interim/summaries/multivariate.pkl')

    # Scores -> DataFrame
    tt_split = pd.read_csv(DATA_DIR + '/raw/summaries/accuracies/singleTrainTest.csv', index_col=0)
    resample_split = pd.read_csv(DATA_DIR + '/raw/summaries/accuracies/resamples.csv', index_col=0)
    save_pickle(tt_split, DATA_DIR + '/interim/summaries/accuracies/train_test_split.pkl')
    save_pickle(resample_split, DATA_DIR + '/interim/summaries/accuracies/resample_split.pkl')
