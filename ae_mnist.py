import tensorflow as tf
from MLkit import tf_networks as nets
from MLkit.tf_math import accuracy

from utils.ae_utils import run_ae
from utils.data_utils import import_data, DataName
from utils.feature_eval import feature_eval_setup

data_name = DataName.MNIST
data, data_test = import_data(data_name)
name = 'AE'
dim_z = 100
mb_size = 128

input_size = [None] + data.dim_X
X = tf.placeholder(tf.float32, shape=[None, data.dim_x])
X__ = tf.reshape(X, shape=[-1] + data.dim_X + [1], name='X__')

with tf.variable_scope('E'):
    Z_logits = nets.conv_only28(X__, dim_z)
    Z = tf.nn.relu(Z_logits)

with tf.variable_scope('G'):
    G_logits = nets.deconv28(Z, is_train=True)
    G_X = tf.nn.sigmoid(G_logits)

loss = tf.reduce_mean(
    tf.nn.l2_loss(
        G_X - X__
    ))

train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
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
       feature_eval=None,
       train=train,
       loss=loss,
       X=X,
       G_X=G_X,
       sess=sess,
       experiment_id=name)
