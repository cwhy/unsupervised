from enum import Enum, auto

from MLkit.dataset import DataSet
from tensorflow.examples.tutorials.mnist import input_data
import os.path as op
from strange_sets.count import CountSquares
import numpy as np


class DataName(Enum):
    MNIST = auto()
    STRANGE = auto()
    PDSST = auto()


def import_data(data_name: DataName):
    if data_name == DataName.MNIST:
        mnist_dir = op.join(op.expanduser('~'), 'Data', 'tf_MNIST')
        mnist_raw = input_data.read_data_sets(mnist_dir, one_hot=True)
        mnist = DataSet(mnist_raw.train.images, mnist_raw.train.labels, dim_X=[28, 28])
        mnist_test = DataSet(mnist_raw.test.images, mnist_raw.test.labels, dim_X=[28, 28])
        return mnist, mnist_test
    elif data_name == DataName.STRANGE:
        strange_cs = CountSquares(flatten_x=True)
        return strange_cs, strange_cs
    elif data_name == DataName.PDSST:
        sst_dir = op.join(op.expanduser('~'), 'Data', 'data_SST')
        sst_raws = []
        for i in range(1, 11):
            file_name = f'screen_blue_CakeBalloon_{i}.npz'
            with np.load(op.join(sst_dir, file_name)) as data:
                sst_raw_i = data['screens']
                sst_raws.append(sst_raw_i)
        sst_raw = np.concatenate(sst_raws, 0)/255
        sst = DataSet(sst_raw, sst_raw[:, 0, 0, 0])
        return sst, sst
