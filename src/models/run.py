from src.data.make_dataset import UcrDataset
from src.models.model import ShapeletNet
from src.models.dataset import PointsDataset, SigletDataset





if __name__ == '__main__':
    ds_names = ['Beef', 'BeetleFly', 'Chlorine']

    for ds_name in ds_names:
        ucr_train, ucr_test = UcrDataset(ds_name=ds_name).get_original_train_test()



