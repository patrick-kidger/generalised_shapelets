import torch
from torch import nn, optim
from sklearn.cluster import KMeans

# # CUDA
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:3" if use_cuda else "cpu")


class PatrickDiscriminator(torch.nn.Module):
    def __init__(self, shapelet_len, mode):
        assert mode in range(1, 10)
        super(PatrickDiscriminator, self).__init__()
        self.shapelet_len = shapelet_len
        self.mode = mode
        self.vector = torch.nn.Parameter(torch.randn(shapelet_len))

    def forward(self, diffs):
        vector = self.vector if self.mode in (1, 2, 3, 4, 5) else self.vector ** 2
        if self.mode == 1:  # very nearly james' version. He adds a bias as well.
            diffs = diffs.abs()
            diffs = (vector * diffs).sum(dim=-1)
            diffs.abs()
        elif self.mode in (2, 3, 6, 8):
            diffs = (vector * diffs).sum(dim=-1)
            diffs = diffs.abs()
        elif self.mode in (4, 5, 7, 9):
            diffs = (vector * diffs).sum(dim=-1)
            diffs = diffs ** 2
        diffs, _ = diffs.min(dim=1)
        if self.mode in (3, 5, 8, 9):
            diffs = diffs.log()
        return diffs


class ShapeletNet(nn.Module):
    """ Generic model for shapelet learning. """
    def __init__(self, num_shapelets, shapelet_len, num_outputs, init_data=None, discriminator='l2'):
        super().__init__()
        self.num_shapelets = num_shapelets
        self.shapelet_len = shapelet_len
        self.num_outputs = num_outputs

        # Setup shapelets
        self.shapelet_init_val = nn.Parameter(torch.ones(1, 1), requires_grad=False)
        self.shapelets_ = nn.Linear(1, self.num_shapelets * self.shapelet_len, bias=False)
        if init_data is not None:
            self._init_shapelets(init_data)
        self.shapelets = lambda: self.shapelets_(self.shapelet_init_val).view(self.num_shapelets, self.shapelet_len)

        # Discriminator for evaluating similarity
        if discriminator == 'l1':
            self.discriminator = lambda diffs: torch.norm(diffs, dim=3, p=1)
        elif discriminator == 'l2':
            self.discriminator = lambda diffs: torch.norm(diffs, dim=3, p=2)
        elif discriminator == 'mlp':
            self.discriminator = nn.Sequential(
                nn.Linear(self.shapelet_len, self.shapelet_len, bias=0),
                nn.ReLU(),
                nn.Linear(self.shapelet_len, 1, bias=0)
            )
        elif discriminator == 'linear':
            self.discriminator = nn.Sequential(
                nn.Linear(self.shapelet_len, 1),
            )
        elif 'patrick' in discriminator:
            mode = int(discriminator[len('patrick'):])
            self.discriminator = PatrickDiscriminator(self.shapelet_len, mode)

        # Classifier on the min value of the discriminator
        self.classifier = nn.Linear(self.num_shapelets, num_outputs, bias=False)

    def _init_shapelets(self, data):
        """ Shapelet initialisation using k-means clustering. """
        # Compute centroids
        data = data.to('cpu')
        kmeans = KMeans(n_clusters=self.num_shapelets)
        kmeans.fit(data.reshape(-1, data.size(2)))
        cluster_centers = kmeans.cluster_centers_

        # Update weights to start at cluster centers
        self.shapelets_.weight.data = torch.Tensor(cluster_centers.reshape(-1, 1))

    def forward(self, x):
        # Get shapelets
        shapelets = self.shapelets()

        # Compute the difference
        diffs = x.unsqueeze(2) - shapelets
        if isinstance(self.discriminator, PatrickDiscriminator):
            min_discrim = self.discriminator(diffs)
        else:
            diffs = torch.abs(diffs)

            # Get min discrimination
            discrim = self.discriminator(diffs).squeeze(-1)
            min_discrim, _ = discrim.min(dim=1)

            # Additional bits
            # min_discrim = torch.log(min_discrim + 1e-5)

        # Apply the classifier
        predictions = self.classifier(min_discrim)

        return predictions



if __name__ == '__main__':
    from jamesshapelets.src.data.make_dataset import UcrDataset
    from jamesshapelets.src.models.dataset import SigletDataset, PointsDataset
    from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score

    # Params
    ds_name = 'ECGFiveDays'
    window_size = 30
    depth = 5
    n_iters = 1000

    print(ds_name)

    # Get datasets
    ucr_train, ucr_test = UcrDataset(ds_name=ds_name).get_original_train_test()
    n_classes = ucr_train.n_classes
    n_outputs = n_classes if n_classes > 2 else n_classes - 1
    train_ds, test_ds = [SigletDataset(x.data, x.labels, depth=depth, ds_length=x.size(1), num_window_sizes=5) for x in (ucr_train, ucr_test)]
    # train_ds, test_ds = [PointsDataset(x.data, x.labels, window_size=window_size) for x in (ucr_train, ucr_test)]
    train_dl = DataLoader(train_ds, batch_size=32)

    num_shapelets = 50
    shapelet_len = train_ds.data.size(2)

    # Setup
    model = ShapeletNet(num_shapelets=num_shapelets, shapelet_len=shapelet_len, num_outputs=n_outputs, init_data=train_ds.data, discriminator='linear')
    model.to(device)
    criterion = nn.CrossEntropyLoss() if n_classes > 2 else nn.BCELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.1)

    # Run
    for iter in range(n_iters):
        losses = []
        model.train()
        for i, (data, labels) in enumerate(train_dl):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(data)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if iter % 50 == 0:
            print('Iter: {} - Loss: {}'.format(iter, np.mean(losses)))

            # Validate
            model.eval()
            with torch.no_grad():
                data, labels = test_ds.data.to(device), test_ds.labels.to(device)
                #
                if len(np.unique(labels)) > 2:
                    preds = torch.argmax(model(data), axis=1)
                    print('Accuracy score: {:.2f}%'.format(100 * accuracy_score(preds, labels)))
                else:
                    preds = model(data)
                    print('Accuracy score: {:.2f}%'.format(100 * accuracy_score((preds > 0.5).int(), labels)))

    # Plot final shapelets
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    shapelets = model.shapelets().detach()
    ax[0].plot(shapelets[1, :])
    ax[1].plot(shapelets[0, :])
    plt.show()





