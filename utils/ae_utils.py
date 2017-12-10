import os.path as op
from itertools import count
from typing import Callable, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from MLkit.dataset import DataSet, Dimensions
from matplotlib import gridspec
from tensorflow import Tensor, Session, Operation

from utils.feature_eval import Number
from utils.setup import out_dir


def get_out_dir(save_id: str):
    return op.join(out_dir, f'{save_id}.png')


def plot_rec_sample(data_iter: Callable[[int], DataSet],
                    dim_X: Dimensions,
                    X_in: Tensor,
                    X_out: Tensor,
                    sess: Session,
                    experiment_id: str,
                    iter_id: str,
                    n_rows: int = 3,
                    _n: int = 9):
    _x = data_iter(_n).x
    s_X, s_rec = sess.run([X_in, X_out],
                          feed_dict={X_in: _x})
    save_id = '_'.join([experiment_id, 'rec', iter_id])
    fig = plt.figure(figsize=(n_rows * 7, n_rows * 3))
    axarr = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
    plot_grid(dim_X, s_X, n_rows, axarr[0], fig)
    plot_grid(dim_X, s_rec, n_rows, axarr[1], fig)
    plt.savefig(get_out_dir(save_id), bbox_inches='tight')
    print(f'Saved: {save_id}')
    plt.close(fig)


def plot_grid(dim_X, samples, _rows, outer_ax, fig):
    gs = gridspec.GridSpecFromSubplotSpec(_rows, _rows,
                                          subplot_spec=outer_ax,
                                          wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        if dim_X[2] == 1:  # Monotone image
            sample = np.reshape(sample, dim_X[:-1])
        else:
            sample = np.reshape(sample, dim_X)
        ax = plt.Subplot(fig, gs[i])
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.imshow(sample, cmap='Greys_r')
        fig.add_subplot(ax)


def default_feeder(data: DataSet,
                   X: Tensor,
                   _size: int):
    x_mb = data.sample(_size).x
    return {X: x_mb}


def run_ae(data: DataSet,
           mb_size: int,
           train: Operation,
           loss: Tensor,
           X: Tensor,
           G_X: Tensor,
           sess: Session,
           experiment_id: str,
           feature_eval: Optional[Callable[[Session], Tuple[Number, Number]]],
           data_feeder: Callable[
               [DataSet, Tensor, int],
               Dict[Tensor, np.ndarray]] = default_feeder,
           max_iter: int = 1000000):
    counter = count(1)
    for it in range(max_iter):
        if it % 10000 == 0:
            iter_id = str(next(counter)).zfill(3)
            plot_rec_sample(data.sample, data.dim_X, X, G_X, sess, experiment_id, iter_id)

            if it % 10000 == 0 and feature_eval is not None:
                fe_loss, fe_acc = feature_eval(sess)
                print(f"Feature Evaluation at {iter_id}: loss -> {fe_loss}, Accuracy -> {fe_acc}")

        data_feed = data_feeder(data, X, mb_size)
        sess.run(train, feed_dict=data_feed)

        if it % 500 == 0:
            data_feed = data_feeder(data, X, mb_size)
            loss_val = sess.run(loss, feed_dict=data_feed)
            print('Iter: {}'.format(it))
            print('Loss: {:.4}'.format(loss_val))
            print()
