import tensorflow as tf
from MLkit import tf_networks as nets

from utils.ae_utils import run_ae, default_feeder
from utils.data_utils import import_data, DataName
from utils.feature_eval import interpolation_setup

data_name = DataName.PDSST
data, data_test = import_data(data_name)
name = 'RVAE'
dim_z = 256
mb_size = 128

input_size = [None] + data.dim_X
X = tf.placeholder(tf.float32, shape=[None, data.dim_x])
X__ = tf.reshape(X, shape=[-1] + data.dim_X, name='X__')

recon_loss = 0
kl_loss = 0
T = 2
X_means = []
Xt = X__
Zs = []
for t in range(T):
    with tf.variable_scope('E') as scope:
        if t != 0:
            scope.reuse_variables()
        tmp_logits = nets.conv80(Xt, 1024, is_train=True)
        Zt, kl_loss_t = nets.get_variational_layer(tmp_logits, dim_z)
        Zs.append(Zt)
        kl_loss += tf.reduce_mean(kl_loss_t)

    with tf.variable_scope('G') as scope:
        if t != 0:
            scope.reuse_variables()
        G_logits = nets.deconv80(Zt, out_channels=2 * data.dim_X[-1], is_train=True)
    Xt_mean = tf.sigmoid(G_logits[:, :, :, :3])
    Xt_logvar = tf.tanh(G_logits[:, :, :, 3:])
    eps = tf.random_normal(shape=tf.shape(Xt_mean))
    Xt = Xt_mean + eps * (tf.exp(Xt_logvar / 2))
    X_means.append(Xt_mean)
    recon_loss_t = tf.reduce_mean(tf.reduce_sum((Xt_mean - X__) ** 2, axis=(1, 2, 3)))
    recon_loss += recon_loss_t

G_X = X_means[0]
loss = recon_loss + kl_loss/T

train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
print([_.name for _ in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])

sess = tf.Session()
interpolation = interpolation_setup(
    X,
    G_X,
    data.dim_X,
    Zs[0],
)

sess.run(tf.global_variables_initializer())
data_feed = default_feeder(data, X, mb_size)
print(sess.run([recon_loss, kl_loss], feed_dict=data_feed))
run_ae(data=data,
       mb_size=mb_size,
       interpolation=interpolation,
       feature_eval=None,
       train=train,
       loss=loss,
       X=X,
       G_X=G_X,
       sess=sess,
       experiment_id=name)
