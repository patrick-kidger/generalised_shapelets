from jamesshapelets.definitions import *
import pandas as pd

s = pd.read_csv('./grabocka_results.csv', index_col=0)
s.set_index('Dataset', inplace=True)
save_pickle(s, DATA_DIR + '/interim/summaries/grabocka.pkl')
