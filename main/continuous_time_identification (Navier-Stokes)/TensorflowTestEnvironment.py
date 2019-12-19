"""
@author: Maziar Raissi
"""
import sys

sys.path.insert(0, '../../Utilities/')

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting import newfig, \
    savefig  # this is the thing in the Utilities subfolder (added above to the sys path) - NOT a 3rd party module
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import pickle
import ttictoc
import tqdm
import TensorboardTools
import sys
import warnings

np.random.seed(1234)
tf.set_random_seed(1234)


def array_extending_insert(array, index, value):
    try:
        array[index]  # just see if the array is long enough. Actual insertion is done in the finally: block
    except IndexError:
        array = np.concatenate((array, np.zeros(1000)))
    finally:
        try:
            array[index] = value
        except IndexError as index_exception:
            raise type(index_exception)(str(index_exception) +
                                        "Custom exception: array extension was insufficiently long."
                                        ).with_traceback(sys.exc_info()[2])

    return array


class PhysicsInformedNN:

    def __getstate__(self):
        odict = self.__dict__.copy()
        variables_to_remove = ['sess', 'optimizer_Adam', 'optimizer', 'train_op_Adam', 'weights', 'biases', 'lambda_1',
                               'lambda_2', 'x_tf', 'y_tf', 't_tf', 'u_tf', 'v_tf', 'u_pred', 'v_pred', 'p_pred',
                               'f_u_pred', 'f_v_pred', 'loss', 'p_at_first_node', 'loss_summary', 'psi_pred']

        for variable_to_remove in variables_to_remove:
            try:
                del odict[variable_to_remove]
            except KeyError:
                warnings.warn('Variable {} not found during pickling of model.'.format(variable_to_remove))
        return odict

    def __setstate__(self, odict):
        self.__dict__.update(odict)

        # support for missing variable in old save files
        self.discover_navier_stokes_parameters = False
        self.p_first_spacetime_node = None
        self.p_at_first_node = 3.141592

        self.finalise_state_setup()

    def finalise_state_setup(self):
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        if self.discover_navier_stokes_parameters:
            # Initialize parameters
            self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
            self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
        else:
            self.lambda_1 = tf.constant([1.0], dtype=tf.float32)
            self.lambda_2 = tf.constant([0.01], dtype=tf.float32)

        # tf placeholders and graph
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False,
                                                     gpu_options=gpu_options))

        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        # self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # These placeholders all have shape [None, 1], as self.x.shape[1] = 1.
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])

        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])

        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred, self.psi_pred, p_at_first_node = \
            self.net_NS(self.x_tf, self.y_tf, self.t_tf)

        if self.p_first_spacetime_node is not None:
            self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                        tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
                        tf.reduce_sum(tf.square(self.f_u_pred)) + \
                        tf.reduce_sum(tf.square(self.f_v_pred)) + \
                        tf.square(p_at_first_node - 10.0)

            print("shape was: {}".format(p_at_first_node.shape))

            # tf.square(self.p_pred[1] - self.p_first_spacetime_node)
        else:
            self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                        tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
                        tf.reduce_sum(tf.square(self.f_u_pred)) + \
                        tf.reduce_sum(tf.square(self.f_v_pred))

        self.loss_summary = tf.summary.scalar("loss", self.loss)

        self.max_optimizer_iterations = 50000  # 50000
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': self.max_optimizer_iterations,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def getMaxOptimizerIterations(self):
        return self.max_optimizer_iterations

    # Initialize the class
    def __init__(self, x, y, t, u, v, layers, p_reference_point, discover_navier_stokes_parameters):
        self.p_at_first_node = 0.0
        self.discover_navier_stokes_parameters = discover_navier_stokes_parameters

        X = np.concatenate([x, y, t], 1)

        self.lb = X.min(0)
        self.ub = X.max(0)

        # The above gives us a list of max and min in each of the 4 dimensions of X.
        # in some cases (e.g. bias term of all ones), we'll get ub=lb, and later we
        # have ub-lb in a denominator - handle this case here.
        for index, (upper, lower) in enumerate(zip(self.ub, self.lb)):
            if upper == lower:
                self.ub[index] = self.ub[index] + 1.0

        self.X = X

        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.t = X[:, 2:3]

        self.p_first_spacetime_node = p_reference_point

        self.u = u
        self.v = v

        self.layers = layers

        self.iteration_counter = 0

        self.finalise_state_setup()

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        # bias_term = tf.fill([tf.shape(x)[0], 1], 1.0)

        psi_and_p = self.neural_net(tf.concat([x, y, t], 1), self.weights, self.biases)

        psi = psi_and_p[:,
              0]  # Seems to be that this is a scalar potential for the velocity, and that is what we predict
        p = psi_and_p[:, 1]  # Pressure field

        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
        f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)

        # run the NN a second time on the first node (at x=1, y=-2, t=0) to get a reference pressure whose value we can
        # target in order to control the pressure field's absolute values;
        psi_and_p_ignore_psi = self.neural_net(tf.constant([1.0, -2.0, 0.0], shape=(1, 3)), self.weights, self.biases)
        self.p_at_first_node = psi_and_p_ignore_psi[:, 1]

        return u, v, p, f_u, f_v, psi, self.p_at_first_node

    def callback(self, loss, lambda_1, lambda_2):
        self.iteration_counter += 1

        self.loss_history = array_extending_insert(self.loss_history, self.loss_history_write_index, loss)

        self.loss_history_write_index += 1

        print('(A) Loss: %.3e, l1: %.3f, l2: %.5f (%f/%f)' % (loss, lambda_1, lambda_2,
                                                              self.iteration_counter, self.max_optimizer_iterations))

    def train(self, nIter, summary_writer, x_predict, y_predict, t_predict):

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                   self.u_tf: self.u, self.v_tf: self.v}

        self.loss_history = np.zeros(
            self.max_optimizer_iterations + 1)  # 1 extra as a marker of where it switches between the optimizers
        self.loss_history_write_index = 0

        start_time = time.time()
        for it in tqdm.tqdm(range(nIter)):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value, loss_summary_retrieved = self.sess.run([self.loss, self.loss_summary], tf_dict)
                self.loss_history = array_extending_insert(self.loss_history, self.loss_history_write_index, loss_value)
                self.loss_history_write_index += 1

                summary_writer.add_summary(loss_summary_retrieved, it)

                if it % 5000 == 0:
                    # Show an image
                    _, _, p_star, _ = self.predict(x_predict, y_predict, t_predict)
                    image = np.reshape(p_star, (-1, 50, 100, 1))
                    summary_image_op = tf.summary.image("Snapshot", image)
                    summary_image = self.sess.run(summary_image_op)
                    summary_writer.add_summary(summary_image)

                if self.discover_navier_stokes_parameters:
                    lambda_1_value = self.sess.run(self.lambda_1)
                    lambda_2_value = self.sess.run(self.lambda_2)
                else:
                    lambda_1_value = -1.0
                    lambda_2_value = -1.0

                reference_node_pressure = self.sess.run(self.p_at_first_node)
                print('It: %d, (B) Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f, %f' %
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed, reference_node_pressure))
                start_time = time.time()

        # just a marker to see where the optimizers switched over
        self.loss_history = array_extending_insert(self.loss_history, self.loss_history_write_index, -1.0)

        self.loss_history_write_index += 1

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.lambda_1, self.lambda_2],
                                loss_callback=self.callback)

    def predict(self, x_star, y_star, t_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        psi_pred = self.sess.run(self.psi_pred, tf_dict)

        return u_star, v_star, p_star, psi_pred

    def getLossHistory(self):
        return self.loss_history


def plot_solution(X_star, u_star, index, title):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

    plt.figure(index)
    plt.pcolor(X, Y, U_star, cmap='jet')
    plt.colorbar()
    plt.title(title)
    plt.savefig(title.replace(" ", "_") + '.png')


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


if __name__ == "__main__":

    tensorboard_log_directory_base = r'.\testlogs'

    # Warning: this assumes that the only contents of the logs directory is subdirs with integer names.
    integer_log_subdir_names = [int(filename) for filename in os.listdir(tensorboard_log_directory_base)]
    if len(integer_log_subdir_names) == 0:
        next_available_integer_for_subdir_name = 1
    else:
        next_available_integer_for_subdir_name = max(integer_log_subdir_names) + 1

    tensorboard_log_directory = '{}\\{}'.format(tensorboard_log_directory_base, next_available_integer_for_subdir_name)
    os.mkdir(tensorboard_log_directory)

    with tf.device("/cpu:0"), TensorboardTools.TensorboardProcess(tensorboard_log_directory_base, 6006):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False,
                                                     gpu_options=gpu_options))

        tf.set_random_seed(1234)
        print(sess.run(tf.constant("hello world!")))
        tf.reset_default_graph()
        tf.set_random_seed(1234)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=False,
                                                gpu_options=gpu_options))
        print(sess.run(tf.truncated_normal((10,10), stddev=0.1)))
