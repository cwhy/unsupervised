import tensorflow as tf
from MLkit import tf_networks as nets
from MLkit.tf_networks import scewl

from utils.ae_utils import run_ae, default_feeder
from utils.data_utils import import_data, DataName
from utils.feature_eval import interpolation_setup, view_dist_setup, view_disentangle_setup

data_name = DataName.PDSST
data, data_test = import_data(data_name)
name = 'AAE'
dim_z = 32
mb_size = 128
lr = 1e-4

input_size = [None] + data.dim_X
X = tf.placeholder(tf.float32, shape=[None, data.dim_x])
X__ = tf.reshape(X, shape=[-1] + data.dim_X, name='X__')

with tf.variable_scope('E'):
    tmp_logits = nets.conv80(X__, dim_z, is_train=True)
    tmp_logits = nets.mini_rnn(tmp_logits, out_dim_t=4, state_size=64)
    Z = nets.dense_net(tmp_logits, [256, dim_z])
    # Z, _ = nets.get_variational_layer(tmp_logits, dim_z)
    # Z = tmp_logits

with tf.variable_scope('G'):
    tmp_logits = nets.mini_rnn(Z, out_dim_t=4, state_size=64)
    tmp_logits = nets.dense_net(tmp_logits, [512],
                                activation_fn=tf.nn.relu,
                                batch_norm=True, is_train=True)
    G_logits = nets.deconv80(tmp_logits, out_channels=data.dim_X[-1], is_train=True)
    G_X = tf.nn.sigmoid(G_logits)


def D(_in, init=False):
    with tf.variable_scope('D', reuse=not init):
        tmp_logits = nets.mini_rnn(_in, out_dim_t=4, state_size=64)
        _out_logit = nets.dense_net(tmp_logits, [512, 32, 1])
    return _out_logit


dynamic_mb_size = tf.shape(Z)[0]
dim_discrete = 2
discrete_shape = (dynamic_mb_size, dim_discrete)
bernoulli_samples = tf.where(tf.random_uniform(discrete_shape) - 0.5 < 0,
                             tf.ones(discrete_shape), tf.zeros(discrete_shape))
normal_samples = tf.random_normal(shape=(dynamic_mb_size, dim_z - dim_discrete))

D_logit_fake = D(Z, init=True)
D_logit_real = D(tf.concat((bernoulli_samples, normal_samples), axis=1))

vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D/')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G/')
vars_E = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'E/')
D_loss_real = tf.reduce_mean(scewl(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(scewl(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(scewl(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake))) - D_loss_fake

recon_loss = tf.reduce_mean(tf.reduce_sum(0.1*tf.abs(G_X - X__) + (G_X - X__) ** 2,
                                          axis=[1, 2, 3])**2)

train_D = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=vars_D)
train_G = tf.train.AdamOptimizer(lr).minimize(recon_loss, var_list=vars_G + vars_E)
train_E = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=vars_E)
print([_.name for _ in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
train = tf.group(train_D, train_E, train_G)

sess = tf.Session()
view_dist = view_dist_setup(X, Z)

view_disentangle = view_disentangle_setup(
    X,
    G_X,
    data.dim_X,
    Z,
)

sess.run(tf.global_variables_initializer())
data_feed = default_feeder(data, X, mb_size)
run_ae(data=data,
       mb_size=mb_size,
       interpolation=None,
       view_disentangle=view_disentangle,
       view_dist=view_dist,
       feature_eval=None,
       train=train,
       loss=recon_loss,
       X=X,
       G_X=G_X,
       sess=sess,
       experiment_id=name)
