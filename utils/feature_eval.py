from typing import Callable, Optional, Tuple, Union

from matplotlib import gridspec
from MLkit.mpl_helper import plt

import tensorflow as tf
from MLkit.dataset import DataSet, Dimensions
from MLkit.tf_networks import dense_net
from tensorflow import Tensor, Session
from tqdm import tqdm
import numpy as np

from utils.ae_utils import plot_grid, get_out_dir, Number


def feature_eval_setup(sess: Session,
                       X: Tensor,
                       Z: Tensor,
                       data_train: DataSet,
                       data_test: DataSet,
                       eval_fn: Callable[[Tensor, Tensor], Tensor],
                       eval_loss_fn: Callable[[Tensor, Tensor], Tensor],
                       supervise_net: Optional[Callable[[Tensor], Tensor]] = None,
                       optimizer: tf.train.Optimizer = (
                               tf.train.RMSPropOptimizer(learning_rate=1e-4)),
                       mb_size: Optional[int] = 128,
                       max_iter: int = 5000,
                       restart_training: bool = True
                       ) -> Callable[[Session], Tuple[Number, Number]]:
    with tf.variable_scope('feature_eval'):
        if supervise_net is not None:
            y_logits = supervise_net(Z)
        else:
            y_logits = dense_net(Z, [256, data_train.dim_y])

    y_hat = tf.sigmoid(y_logits)
    y = tf.placeholder(tf.float32, [None] + data_train.dim_Y)
    eval_loss = tf.reduce_mean(eval_loss_fn(y_logits, y))
    eval_result = eval_fn(y_hat, y)
    vars_fteval = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='feature_eval')
    train = optimizer.minimize(eval_loss, var_list=vars_fteval)
    eval_vars_initializer = tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='feature_eval'))
    sess.run(eval_vars_initializer)

    def feature_eval(_sess: Session) -> Tuple[Number, Number]:
        if restart_training:
            _sess.run(eval_vars_initializer)
        for _ in tqdm(range(max_iter)):
            if mb_size is not None:
                _mb = data_train.sample(mb_size)
            else:
                _mb = data_train
            data_feed = {X: _mb.x, y: _mb.y}
            _sess.run(train, feed_dict=data_feed)
        data_feed = {X: data_test.x, y: data_test.y}
        val_eval_loss = _sess.run(eval_loss, feed_dict=data_feed)
        val_eval = _sess.run(eval_result, feed_dict=data_feed)
        return val_eval_loss, val_eval

    return feature_eval


def interpolation_setup(X_in: Tensor,
                        X_out: Tensor,
                        dim_X: Dimensions,
                        Z: Tensor,
                        X_pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                        n_figs: Optional[int] = 16,
                        n_rows: int = 4
                        ) -> Callable[[Callable[[int], DataSet],
                                       Session, str, str], None]:
    def interpolation(data_iter: Callable[[int], DataSet],
                      sess: Session,
                      experiment_id: str,
                      iter_id: str):
        if X_pairs is not None:
            for xp in X_pairs:
                assert xp.shape[0] == 1
                assert len(xp.shape) == 2
            _x = np.hstack(X_pairs)
        else:
            _x = data_iter(2).x
        s_Zs = sess.run(Z, feed_dict={X_in: _x})
        s_Z0 = s_Zs[[0], :]
        s_Z1 = s_Zs[[1], :]
        d = (s_Z1 - s_Z0) / 15
        zs = s_Z0 + (d.T @ np.arange(n_figs)[np.newaxis, :]).T
        s_Xs = sess.run(X_out, feed_dict={Z: zs})
        save_id = '_'.join([experiment_id, 'interp', iter_id])
        fig = plt.figure(figsize=(n_rows * 3, n_rows * 3))
        axarr = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)
        plot_grid(dim_X, s_Xs, n_rows, axarr[0], fig)
        plt.savefig(get_out_dir(save_id), bbox_inches='tight')
        print(f'Saved interpolation: {save_id}')
        plt.close(fig)

    return interpolation


def view_dist_setup(X_in: Tensor, Z: Tensor, sample_size: int = 128):
    def view_z_dist(data_iter: Callable[[int], DataSet],
                    sess: Session,
                    experiment_id: str,
                    iter_id: str):
        Z_vals = sess.run(Z, feed_dict={X_in: data_iter(sample_size).x})
        save_id = '_'.join([experiment_id, 'dist', iter_id])
        fig, ax = plt.subplots()
        cax = ax.matshow(Z_vals, vmin=-3, vmax=3, cmap=plt.get_cmap('jet'))
        cbar = fig.colorbar(cax)
        plt.savefig(get_out_dir(save_id), bbox_inches='tight')
        print(f'Saved distribution histograms: {save_id}')
        plt.close(fig)

    return view_z_dist


def view_disentangle_setup(X_in: Tensor,
                           X_out: Tensor,
                           dim_X: Dimensions,
                           Z: Tensor,
                           X_base: np.ndarray = None,
                           z_dim_selected: int=None,
                           n_figs: Optional[int] = 16,
                           n_rows: int = 2
                           ) -> Callable[[Callable[[int], DataSet],
                                          Session, str, str], None]:
    def view_disentangle(data_iter: Callable[[int], DataSet],
                         sess: Session,
                         experiment_id: str,
                         iter_id: str):
        if X_base is not None:
            assert X_base.shape[0] == 1
            assert len(X_base.shape) == 2
            _x = X_base
        else:
            _x = data_iter(1).x
        if z_dim_selected is not None:
            z_dim = z_dim_selected
        else:
            z_dim = np.random.choice(Z.shape[1])
        s_Z = sess.run(Z, feed_dict={X_in: _x})
        d = np.zeros((n_figs, Z.shape[1]))
        d[:, z_dim] = np.linspace(-3, 3, n_figs)
        z_val = s_Z[0, z_dim]
        s_Z[0, z_dim] = 0
        zs = d + s_Z
        s_Xs = sess.run(X_out, feed_dict={Z: zs})
        save_id = '_'.join([experiment_id, 'disent', iter_id])
        fig = plt.figure(figsize=(np.ceil(n_figs/n_rows) * 3, n_rows * 3))
        axarr = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)
        plot_grid(dim_X, s_Xs, n_rows, axarr[0], fig)
        fig.suptitle(f'$Z_{{{z_dim}}}$ from -3 to 3, sample has $Z_{{{z_dim}}}={z_val}$')
        plt.savefig(get_out_dir(save_id), bbox_inches='tight')
        print(f'Saved disentangle: {save_id}')
        plt.close(fig)

    return view_disentangle
