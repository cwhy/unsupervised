import tensorflow as tf
from MLkit import tf_networks as nets
from MLkit.tf_math import accuracy

from utils.ae_utils import run_ae
from utils.data_utils import import_data, DataName
from utils.feature_eval import feature_eval_setup, interpolation_setup

data_name = DataName.MNIST
data, data_test = import_data(data_name)
name = 'RVAE'
dim_z = 8
mb_size = 128

input_size = [None] + data.dim_X
X = tf.placeholder(tf.float32, shape=[None, data.dim_x])
X__ = tf.reshape(X, shape=[-1] + data.dim_X + [1], name='X__')

Xts = [X__]
kl_loss = 0
recon_loss = 0
T = 3
for t in range(T):
    with tf.variable_scope('E') as scope:
        if t != 0:
            scope.reuse_variables()
        tmp_logits = nets.conv_only28(Xts[t], 1024, is_train=True)
        Z, kl_loss_t = nets.get_variational_layer(tmp_logits, dim_z)
        kl_loss += kl_loss_t

    with tf.variable_scope('G') as scope:
        if t != 0:
            scope.reuse_variables()
        G_logits = nets.deconv28(Z, is_train=True)
        G_X = tf.nn.sigmoid(G_logits)
    Xts.append(G_X)
    recon_loss_t = tf.nn.l2_loss(G_X - X__)
    recon_loss += recon_loss_t

# recon_loss = tf.nn.l2_loss(Xt - X__)
# recon_loss = tf.reduce_mean(tf.abs(G_X - X__), 1)
loss = tf.reduce_mean(recon_loss + kl_loss/T)

train = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss)
print([_.name for _ in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])

sess = tf.Session()
feature_eval = feature_eval_setup(sess, X, Z,
                                  data.sample(1000),
                                  data_test.sample(100),
                                  accuracy, nets.scewl,
                                  max_iter=1000)
sess.run(tf.global_variables_initializer())
run_ae(data=data,
       mb_size=mb_size,
       feature_eval=feature_eval,
       interpolation=None,
       train=train,
       loss=loss,
       X=X,
       G_X=Xts[-1],
       sess=sess,
       experiment_id=name)
