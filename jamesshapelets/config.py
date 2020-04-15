from jamesshapelets.src.data.dicts import learning_ts_shapelets


grabocka = {
        # 'ds_name': learning_ts_shapelets[0:10],
        'ds_name': ['Coffee', 'ECGFiveDays'],
        'path_tfm': ['signature'],

        'num_shapelets': [50],
        'num_window_sizes': [5],
        'max_window': [1000],
        'depth': [5],

        'discriminator': ['linear'],

        'max_epochs': [100],
        'lr': [1e-1],
}