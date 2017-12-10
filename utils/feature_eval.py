from typing import Callable, Optional, Tuple, Union

from tqdm import tqdm
import tensorflow as tf
from MLkit.dataset import DataSet
from MLkit.tf_networks import dense_net
from tensorflow import Tensor, Operation, Session

Number = Union[int, float]


def feature_eval_setup(sess: Session,
                       X: Tensor,
                       Z: Tensor,
                       data: DataSet,
                       eval_fn: Callable[[Tensor, Tensor], Tensor],
                       eval_loss_fn: Callable[[Tensor, Tensor], Tensor],
                       supervise_net: Optional[Callable[[Tensor], Tensor]] = None,
                       minimize: Callable[[Tensor], Operation] = (
                               tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize),
                       mb_size: Optional[int] = 128,
                       max_iter: int = 1000,
                       restart_training: bool = True
                       ) -> Callable[[Session], Tuple[Number, Number]]:
    with tf.variable_scope('feature_eval'):
        if supervise_net is not None:
            y_logits = supervise_net(Z)
        else:
            y_logits = dense_net(Z, [256, data.dim_y])

    y_hat = tf.sigmoid(y_logits)
    y = tf.placeholder(tf.float32, [None, data.dim_y])
    eval_loss = tf.reduce_mean(eval_loss_fn(y_logits, y))
    eval_result = eval_fn(y_hat, y)
    train = minimize(eval_loss)
    eval_vars_initializer = tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='feature_eval'))
    sess.run(eval_vars_initializer)

    def feature_eval(_sess: Session) -> Tuple[Number, Number]:
        if restart_training:
            _sess.run(eval_vars_initializer)
        for _ in tqdm(range(max_iter)):
            if mb_size is not None:
                _mb = data.sample(mb_size)
            else:
                _mb = data
            data_feed = {X: _mb.x, y: _mb.y}
            _sess.run(train, feed_dict=data_feed)
        data_feed = {X: data.x, y: data.y}
        val_eval_loss = _sess.run(eval_loss, feed_dict=data_feed)
        val_eval = _sess.run(eval_result, feed_dict=data_feed)
        return val_eval_loss, val_eval

    return feature_eval
