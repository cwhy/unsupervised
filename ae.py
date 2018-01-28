import tensorflow as tf
from MLkit import tf_networks as nets

from utils.ae_utils import run_ae
from utils.data_utils import mnist, mnist_test

data_name = 'mnist'
data, data_test = mnist, mnist_test
name = 'AE'
mb_size = 128


def elu(x):
    return tf.maximum(x, 0) + tf.exp(tf.minimum(x, 0)) - 1


X = tf.placeholder(tf.float32, shape=[None, data.dim_x])

G_logits = nets.dense_net(X, [1000, 500, 250, 30, 250, 500, 1000, data.dim_x],
                          activation_fn=elu,
                          batch_norm=False)
G_X = tf.nn.sigmoid(G_logits)


loss = tf.reduce_mean(tf.nn.l2_loss(G_X - X)/data.dim_x)
train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
run_ae(data=data,
       mb_size=mb_size,
       feature_eval=None,
       train=train,
       loss=loss,
       X=X,
       G_X=G_X,
       sess=sess,
       experiment_id=name)
