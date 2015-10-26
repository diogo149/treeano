"""
functions that download data, and return 3 dicts (corresponding
to train/valid/test splits) each with keys "x" and "y"
"""
import os
import subprocess
import urllib
import cPickle
import numpy as np
import sklearn.datasets
import sklearn.cross_validation
import theano
import theano.tensor as T
import treeano

fX = theano.config.floatX


def mnist(random_state=42):
    """
    x is in [0, 1] with shape (b, 1, 28, 28) and dtype floatX
    y is an int32 vector in range(10)
    """
    raw = sklearn.datasets.fetch_mldata('MNIST original')
    # rescaling to [0, 1] instead of [0, 255]
    x = raw['data'].reshape(-1, 1, 28, 28).astype(fX) / 255.0
    y = raw['target'].astype("int32")
    # NOTE: train data is initially in order of 0 through 9
    x1, x2, y1, y2 = sklearn.cross_validation.train_test_split(
        x[:60000],
        y[:60000],
        random_state=random_state,
        test_size=10000)
    train = {"x": x1, "y": y1}
    valid = {"x": x2, "y": y2}
    # NOTE: test data is in order of 0 through 9
    test = {"x": x[60000:], "y": y[60000:]}
    return train, valid, test


def cifar10(random_state=42, base_dir="~/cifar10"):
    """
    x is in [0, 1] with shape (b, 3, 32, 32) and dtype floatX
    y is an int32 vector in range(10)
    """
    URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    base_dir = os.path.expanduser(base_dir)
    batch_dir = os.path.join(base_dir, "cifar-10-batches-py")
    test_batch = os.path.join(batch_dir, "test_batch")
    if not os.path.isfile(test_batch):
        tar_gz = os.path.join(base_dir, "cifar-10-python.tar.gz")
        try:
            os.mkdir(base_dir)
        except OSError:
            pass
        if not os.path.isfile(tar_gz):
            print("Downloading {} to {}".format(URL, tar_gz))
            urllib.urlretrieve(URL, tar_gz)
        subprocess.call(["tar", "xvzf", tar_gz, "-C", base_dir])

    def read_batch(filename):
        with open(filename, 'rb') as f:
            raw = cPickle.load(f)
        x = raw["data"].reshape(-1, 3, 32, 32).astype(fX) / 255.0
        y = np.array(raw["labels"], dtype="int32")
        return x, y

    # read test data
    test_x, test_y = read_batch(test_batch)
    test = {"x": test_x, "y": test_y}
    # read train+valid data
    xs, ys = [], []
    for i in range(1, 6):
        x, y = read_batch(os.path.join(batch_dir, "data_batch_%d" % i))
        xs.append(x)
        ys.append(y)
    # combine train+valid data
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    # split train and valid
    x1, x2, y1, y2 = sklearn.cross_validation.train_test_split(
        x,
        y,
        random_state=random_state,
        test_size=10000)
    train = {"x": x1, "y": y1}
    valid = {"x": x2, "y": y2}
    return train, valid, test
