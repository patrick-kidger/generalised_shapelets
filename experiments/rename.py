


import os

folder = './results/uea_hyperparameter_search'
files = os.listdir(folder)
for file in files:
    new_file = file.replace('NATOPS-NATOPS', 'NATOPS')
    os.rename(folder + '/' + file, folder + '/' + new_file)