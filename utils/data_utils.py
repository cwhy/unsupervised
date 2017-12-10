from MLkit.dataset import DataSet
from tensorflow.examples.tutorials.mnist import input_data
import os.path as op


mnist_dir = op.join(op.expanduser('~'), 'Data', 'tf_MNIST')
mnist_raw = input_data.read_data_sets(mnist_dir, one_hot=True)
mnist = DataSet(mnist_raw.train.images, mnist_raw.train.labels, dim_X=[28, 28, 1])
