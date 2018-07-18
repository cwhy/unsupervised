import tensorflow as tf
from MLkit import tf_networks as nets
from MLkit.tf_math import accuracy

from utils.ae_utils import run_ae, default_feeder
from utils.data_utils import import_data, DataName
from utils.feature_eval import interpolation_setup, view_dist_setup, view_disentangle_setup
from utils.feature_eval import feature_eval_setup

data_name = DataName.STRANGE
data, data_test = import_data(data_name)
name = 'Rel-VAE'

dim_z = 32
mb_size = 128

input_size = [None] + data.dim_X
X = tf.placeholder(tf.float32, shape=[None, data.dim_x])
X__ = tf.reshape(X, shape=[-1] + data.dim_X + [1], name='X__')


def conv28_ch(x__: tf.Tensor) -> tf.Tensor:
    # In: 28x28 <= shape <= 32x32
    net = tf.layers.conv2d(x__, 16, 5, activation=tf.nn.relu, name='conv1')
    net = tf.layers.max_pooling2d(net, 2, 2, name='pool1')
    net = tf.layers.conv2d(net, 64, 5, activation=tf.nn.relu, name='conv2')
    net = tf.layers.max_pooling2d(net, 2, 2, name='pool2')
    print(net.shape)
    # 15x15
    return net


def get_R(net_in):
    with tf.variable_scope('R'):
        n_objects = int(net.shape[-2])
        n_relations = n_objects * (n_objects - 1) / 2
        attn_in = tf.reshape(net_in, [-1, net_in.shape[-1] * n_objects])
        attn = tf.sigmoid(nets.dense_net(attn_in, [1024, 1024, n_relations]))
        R_Es = []
        counter = 0
        for i in range(n_objects):
            for j in range(i):
                print(i, j)
                with tf.variable_scope('R_R', reuse=tf.AUTO_REUSE):
                    pair = tf.concat([net_in[:, i, :], net_in[:, j, :]], axis=1)
                    attn_ij = tf.expand_dims(attn[:, counter], -1)
                    R_Es.append(attn_ij * nets.dense_net(pair, [1024, 512, 512]))
                counter += 1
        return tf.add_n(R_Es, 'R_E') / n_relations


with tf.variable_scope('E'):
    net = conv28_ch(X__)
    n_objects = int(net.shape[1] * net.shape[2])
    net = tf.reshape(net, [-1, n_objects, net.shape[-1]])
    R_E = get_R(net)

    Z, kl_losses = nets.get_variational_layer(R_E, dim_z)

with tf.variable_scope('G'):
    n_objects = 8
    dim_objects = 64
    net = nets.dense_net(Z, [n_objects * dim_objects])
    net = tf.reshape(net, (-1, n_objects, dim_objects))
    R_G = get_R(net)
    G_logits = nets.deconv28(R_G, is_train=True)
    G_X = tf.nn.sigmoid(G_logits)

sum_dims = list(range(1, 2 + len(data.dim_X)))
recon_loss = tf.reduce_mean(tf.reduce_sum((G_X - X__) ** 2, sum_dims), 0)
# recon_loss = tf.reduce_mean(tf.abs(G_X - X__), 1)
beta = 0.0001
loss = beta * dim_z * tf.reduce_mean(kl_losses) + recon_loss

train = tf.train.AdamOptimizer(learning_rate=1e-7).minimize(loss)
print([_.name for _ in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])

interpolation = interpolation_setup(
    X,
    G_X,
    data.dim_X,
    Z,
)

view_dist = view_dist_setup(X, Z)
view_disentangle = view_disentangle_setup(
    X,
    G_X,
    data.dim_X,
    Z,
)

sess = tf.Session()
feature_eval = feature_eval_setup(sess, X, Z,
                                  data.sample(1000),
                                  data_test.sample(100),
                                  accuracy, nets.scewl,
                                  max_iter=1000)
sess.run(tf.global_variables_initializer())
data_feed = default_feeder(data, X, mb_size)
print(sess.run([recon_loss, tf.reduce_mean(kl_losses)], feed_dict=data_feed))
run_ae(data=data,
       mb_size=mb_size,
       interpolation=interpolation,
       feature_eval=feature_eval,
       view_dist=view_dist,
       view_disentangle=view_disentangle,
       train=train,
       loss=loss,
       X=X,
       G_X=G_X,
       sess=sess,
       experiment_id=name,
       max_iter=10000000)
