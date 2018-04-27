import os.path as op
from typing import Callable, Dict, Tuple, Optional, Union, List

from matplotlib import gridspec
from MLkit.mpl_helper import plt
import numpy as np
from MLkit.dataset import DataSet, Dimensions
from tensorflow import Tensor, Session, Operation

from utils.setup import out_dir
import time

Number = Union[int, float]


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
    n_samples = samples.shape[0]
    n_cols = int(np.ceil(n_samples/_rows))
    gs = gridspec.GridSpecFromSubplotSpec(_rows, n_cols,
                                          subplot_spec=outer_ax,
                                          wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        if len(dim_X) == 2:  # Monotone image
            sample = np.reshape(sample, dim_X)
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


def get_timer_cond(t):
    time_0 = time.time()

    def timer_cond(it):
        nonlocal time_0
        if time.time() - time_0 > t:
            time_0 = time.time()
            return True
        elif it == 0:
            return True
        else:
            return False

    return timer_cond


def run_ae(data: DataSet,
           mb_size: int,
           train: Operation,
           loss: Tensor,
           X: Tensor,
           G_X: Tensor,
           sess: Session,
           experiment_id: str,
           feature_eval: Optional[Callable[[Session], Tuple[Number, Number]]],
           interpolation: Optional[Callable[[Callable[[int], DataSet],
                                             Session, str, str], None]],
           view_dist: Optional[Callable[[Callable[[int], DataSet],
                                         Session, str, str], None]],
           view_disentangle: Optional[Callable[[Callable[[int], DataSet],
                                             Session, str, str], None]],
           data_feeder: Callable[
               [DataSet, Tensor, int],
               Dict[Tensor, np.ndarray]] = default_feeder,
           max_iter: int = 100000) -> None:
    sample_cond = get_timer_cond(50)
    feature_eval_cond = get_timer_cond(100)
    interpolation_cond = get_timer_cond(90)
    print_cond = get_timer_cond(30)
    view_z_cond = get_timer_cond(150)
    view_disentangle_cond = get_timer_cond(100)

    def get_iter_id(_i):
        return str(_i).zfill(int(np.log10(max_iter) + 1))

    # def print_cond(i):
    #     return i % 500 == 0

    # counter = count(1)
    for it in range(max_iter):
        if sample_cond(it):
            # iter_id = str(next(counter)).zfill(3)
            plot_rec_sample(data.sample, data.dim_X, X, G_X,
                            sess, experiment_id, get_iter_id(it))

        if feature_eval_cond(it) and feature_eval is not None:
            fe_loss, fe_acc = feature_eval(sess)
            print(f"Feature Evaluation at {get_iter_id(it)}: loss -> {fe_loss}, Accuracy -> {fe_acc}")

        if interpolation_cond(it) and interpolation is not None:
            interpolation(data.sample, sess, experiment_id, get_iter_id(it))

        if view_disentangle_cond(it) and view_disentangle is not None:
            view_disentangle(data.sample, sess, experiment_id, get_iter_id(it))

        if view_z_cond(it) and view_dist is not None:
            view_dist(data.sample, sess, experiment_id, get_iter_id(it))

        data_feed = data_feeder(data, X, mb_size)
        sess.run(train, feed_dict=data_feed)

        if print_cond(it):
            data_feed = data_feeder(data, X, mb_size)
            loss_val = sess.run(loss, feed_dict=data_feed)
            print('Iter: {}'.format(it))
            print('Loss: {0:.4f}'.format(loss_val))
            print()
