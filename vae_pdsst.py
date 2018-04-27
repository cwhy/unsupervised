import tensorflow as tf
from MLkit import tf_networks as nets

from utils.ae_utils import run_ae, default_feeder
from utils.data_utils import import_data, DataName
from utils.feature_eval import interpolation_setup

data_name = DataName.PDSST
data, data_test = import_data(data_name)
name = 'VAE'
dim_z = 256
mb_size = 128

input_size = [None] + data.dim_X
X = tf.placeholder(tf.float32, shape=[None, data.dim_x])
X__ = tf.reshape(X, shape=[-1] + data.dim_X, name='X__')

with tf.variable_scope('E'):
    tmp_logits = nets.conv80(X__, 1024, is_train=True)
    Z, kl_losses = nets.get_variational_layer(tmp_logits, dim_z)

with tf.variable_scope('G'):
    G_logits = nets.deconv80(Z, out_channels=data.dim_X[-1], is_train=True)
    G_X = tf.nn.sigmoid(G_logits)

recon_loss = tf.reduce_mean((G_X - X__)**2)
# recon_loss = tf.reduce_mean(tf.abs(G_X - X__), 1)
loss = 0.001*tf.reduce_mean(kl_losses) + recon_loss

train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
print([_.name for _ in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])

sess = tf.Session()
interpolation = interpolation_setup(
    sess,
    X,
    G_X,
    data.dim_X,
    Z,
)

sess.run(tf.global_variables_initializer())
data_feed = default_feeder(data, X, mb_size)
print(sess.run([recon_loss, tf.reduce_mean(kl_losses)], feed_dict=data_feed))
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
