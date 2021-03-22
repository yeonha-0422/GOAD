import scipy.io
import numpy as np
import pandas as pd
import torchvision.datasets as dset
import os

class Data_Loader:

    def __init__(self, n_trains=None):
        self.n_train = n_trains

    def norm_kdd_data(self, train_real, val_real, val_fake, cont_indices):
        symb_indices = np.delete(np.arange(train_real.shape[1]), cont_indices)
        mus = train_real[:, cont_indices].mean(0)
        sds = train_real[:, cont_indices].std(0)
        sds[sds == 0] = 1

        def get_norm(xs, mu, sd):
            bin_cols = xs[:, symb_indices]
            cont_cols = xs[:, cont_indices]
            cont_cols = np.array([(x - mu) / sd for x in cont_cols])
            return np.concatenate([bin_cols, cont_cols], 1)

        train_real = get_norm(train_real, mus, sds)
        val_real = get_norm(val_real, mus, sds)
        val_fake = get_norm(val_fake, mus, sds)
        return train_real, val_real, val_fake


    def norm_data(self, train_real, val_real, val_fake):
        mus = train_real.mean(0)
        sds = train_real.std(0)
        sds[sds == 0] = 1

        def get_norm(xs, mu, sd):
            return np.array([(x - mu) / sd for x in xs])

        train_real = get_norm(train_real, mus, sds)
        val_real = get_norm(val_real, mus, sds)
        val_fake = get_norm(val_fake, mus, sds)
        return train_real, val_real, val_fake

    def norm(self, data, mu=1):
        return 2 * (data / 255.) - mu

    def get_dataset(self, dataset_name, c_percent=None, true_label=1):
        if dataset_name == 'MetaDescription':
            return self.MetaDescription_train_valid_data()
        if dataset_name == 'CIDDS':
            return self.CIDDS_train_valid_data()



    def MetaDescription_train_valid_data(self):
        df = pd.read_csv("data/MetaDescription.csv")
        X = df.iloc[::2, 1:16]
        X = X.to_numpy()
        Y = df.iloc[::2, 16]
        Y = np.where(Y.str.contains('normal'), 1, 0)
        samples=X
        labels=Y
        norm_samples=samples[labels == 1]
        anom_samples=samples[labels == 0]
        n_train = len(norm_samples) // 2
        x_train = norm_samples[:n_train]  # 226 train

        val_real = norm_samples[n_train:]
        val_fake = anom_samples
        return self.norm_data(x_train, val_real, val_fake)



    def CIDDS_train_valid_data(self):
        df = pd.read_csv("data/CIDDS.csv")
        X = df.iloc[:, 1:6]
        X = X.to_numpy()
        Y = df.iloc[:, 6]
        Y = np.where(Y.str.contains('normal'), 1, 0)
        samples = X
        labels = Y
        norm_samples = samples[labels == 1]
        anom_samples = samples[labels == 0]
        n_train = len(norm_samples) // 2
        x_train = norm_samples[:n_train]  # 226 train

        val_real = norm_samples[n_train:]
        val_fake = anom_samples
        return self.norm_data(x_train, val_real, val_fake)

