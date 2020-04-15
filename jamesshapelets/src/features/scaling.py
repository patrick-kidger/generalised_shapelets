import torch
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TrickScaler(TransformerMixin):
    """ Tricks an sklearn scaler so that it uses the correct dimensions."""
    def __init__(self, scaler=''):
        if scaler == 'stdsc':
            self.scaler = StandardScaler()
        elif scaler == 'mms':
            self.scaler = MinMaxScaler()

    def _trick(self, X):
        return X.reshape(-1, X.shape[2])

    def _untrick(self, X, shape):
        return X.reshape(shape)

    def fit(self, X, y=None):
        self.scaler.fit(self._trick(X), y)
        return self

    def transform(self, X):
        X_tfm = self.scaler.transform(self._trick(X))
        return torch.Tensor(self._untrick(X_tfm, X.shape))


if __name__ == '__main__':
    import torch
    from jamesshapelets.src.data.make_dataset import UcrDataset
    dataset = UcrDataset(ds_name='LSST', multivariate=True)

    a = TrickScaler().fit_transform(dataset.data)
    import matplotlib.pyplot as plt
    plt.plot(dataset.data[0, :, 0])
    plt.show()
    plt.plot(a[0, :, 0])
    plt.show()


