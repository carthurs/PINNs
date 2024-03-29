"""
@author: Maziar Raissi, Chris Arthurs
"""
import sys

import os
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
import pickle
import ttictoc
import tqdm
import TensorboardTools
import sys
import warnings
import VtkDataReader
import BoundaryConditionCodes as BC
import SimulationParameterManager as SPM
import logging
import ConfigManager
import Meshing
import ActiveLearningUtilities

# np.random.seed(1234)
# tf.set_random_seed(1234)


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

def log_message(message):
    logger = logging.getLogger('SelfTeachingDriver')
    logger.info(message)

class PhysicsInformedNN:
    def __getstate__(self):
        odict = self.__dict__.copy()
        variables_to_remove = ['sess', 'optimizer_Adam', 'train_op_Adam', 'weights', 'biases', 'lambda_1',
                               'lambda_2', 'x_tf', 'y_tf', 't_tf', 'u_tf', 'v_tf', 'r_tf', 'u_pred',
                               'v_pred', 'p_pred',
                               'f_u_pred', 'f_v_pred', 'loss', 'predicted_ref_node_pressures', 'loss_summary', 'psi_pred',
                               'loss_velocity_summary', 'loss_ns_summary', 'loss_navier_stokes', 'loss_velocity',
                               'loss_pressure_node', 'other_summary_scalars', 'loss_pieces_out',
                               'loss_boundary_conditions', 'max_optimizer_iterations', 'bc_codes_tf',
                               'loss_bc_summary', 'loss_pressure_node_summary']

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
        # self.p_reference_point = None
        self.predicted_ref_node_pressures = 3.141592

        self._finalise_state_setup()

    def _finalise_state_setup(self):
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(self.layers)

        if self.discover_navier_stokes_parameters:
            # Initialize parameters
            self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
            self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
        else:
            self.lambda_1 = tf.constant([self.true_density], dtype=tf.float32)
            self.lambda_2 = tf.constant([self.true_viscosity], dtype=tf.float32)

        # tf placeholders and graph
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False,
                                                     gpu_options=gpu_options))

        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        # self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # These placeholders all have shape [None, 1], as self.x.shape[1] = 1.
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]], name='x_placeholder')
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]], name='y_placeholder')
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]], name='t_placeholder')
        self.r_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]], name='r_placeholder')

        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]], name='u_placeholder')
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]], name='v_placeholder')

        self.bc_codes_tf = tf.placeholder(tf.float32, shape=[None], name='bc_codes_placeholder')

        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, \
            self.f_v_pred, self.psi_pred, self.predicted_ref_node_pressures, self.loss_pieces_out =\
            self.net_NS(self.x_tf, self.y_tf, self.t_tf, self.r_tf)

        inflow_condition = lambda y: (10.0-y)*y/25.0 * self.t_tf
        zeros = self.u_pred * 0.0  # stupid, I know, but I can't work out how to get the right shape otherwise
        # self.loss_boundary_conditions = tf.reduce_sum(zeros)

        self.loss_boundary_conditions = tf.reduce_sum(tf.square(
                                            tf.where(tf.less(tf.abs(self.bc_codes_tf - BC.Codes.INFLOW), 0.0001),
                                                           self.u_pred - inflow_condition(self.y_tf), zeros)
                                        )) + \
                                         tf.reduce_sum(
                                             tf.where(tf.less(tf.abs(self.bc_codes_tf - BC.Codes.NOSLIP), 0.0001),
                                                      tf.square(self.u_pred) + tf.square(self.v_pred), zeros))
        # lines are:
        # 1) inflow condition satisfaction on u
        # 2) lower boundary noslip satisfaction
        # 3) upper boundary noslip satisfaction
        # self.loss_boundary_conditions = tf.reduce_sum(
        #     tf.square(tf.where(tf.math.less(self.x_tf, 0.0001), self.u_pred - inflow_condition(self.y_tf), zero))) + \
        #                                 tf.reduce_sum(tf.square(
        #                                     tf.where(tf.math.less(self.y_tf, 0.0001), self.u_pred + self.v_pred,
        #                                              zero))) + \
        #                                 tf.reduce_sum(tf.square(
        #                                     tf.where(tf.math.greater(self.y_tf, 9.999), self.u_pred + self.v_pred,
        #                                              zero)))

        config_manager = ConfigManager.ConfigManager()
        navier_stokes_loss_scaling = config_manager.get_navier_stokes_loss_scaling()
        pressure_node_loss_scaling = config_manager.get_pressure_node_loss_scaling()

        self.loss_velocity = tf.reduce_sum(tf.square(tf.where(tf.math.equal(self.u_tf, tf.constant(-1.0)), zeros, self.u_tf - self.u_pred))) + \
                             tf.reduce_sum(tf.square(tf.where(tf.math.equal(self.v_tf, tf.constant(-1.0)), zeros, self.v_tf - self.v_pred)))
        self.loss_navier_stokes = tf.reduce_sum(tf.square(navier_stokes_loss_scaling*self.f_u_pred)) + \
                                  tf.reduce_sum(tf.square(navier_stokes_loss_scaling*self.f_v_pred))

        if self.ref_nodes_with_true_pressures is not None:
            difference_sum = 0
            for node_index in range(len(self.ref_nodes_with_true_pressures)):
                difference_sum += tf.square(self.predicted_ref_node_pressures[node_index] -
                                            self.ref_nodes_with_true_pressures[node_index][4])

            self.loss_pressure_node = tf.squeeze(difference_sum * pressure_node_loss_scaling**2)

            self.loss_pressure_node_summary = tf.summary.scalar("loss_pressure_node",
                                                                self.loss_pressure_node / (pressure_node_loss_scaling**2))

            # loss_t_gradient = tf.reduce_sum(tf.square(navier_stokes_loss_scaling*psi_t_pred)) +\
            #                   tf.reduce_sum(tf.square(navier_stokes_loss_scaling*p_t_pred))

            # if self.u_tf[0] == -1 and self.v_tf[0] == -1:
            #     # Navier-Stokes loss only here, as the -1s indicate we have no data here
            #     self.loss = self.loss_navier_stokes + loss_t_gradient
            # else:
            self.loss = self.loss_velocity + self.loss_navier_stokes + \
                        self.loss_pressure_node + self.loss_boundary_conditions * navier_stokes_loss_scaling**2
        else:
            self.loss = self.loss_velocity + self.loss_navier_stokes + self.loss_boundary_conditions * navier_stokes_loss_scaling**2

        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.loss_velocity_summary = tf.summary.scalar("loss_velocity", self.loss_velocity)
        self.loss_ns_summary = tf.summary.scalar("loss_navier_stokes",
                                              self.loss_navier_stokes/(navier_stokes_loss_scaling**2))
        self.loss_bc_summary = tf.summary.scalar("loss_bc", self.loss_boundary_conditions)

        # self.loss_pieces_out = [u * v_x, v * v_y, p_y, v_xx, v_yy]
        self.other_summary_scalars = []
        self.other_summary_scalars.append(tf.summary.scalar("u*v_x", tf.reduce_mean(self.loss_pieces_out[0])))
        self.other_summary_scalars.append(tf.summary.scalar("v*v_y", tf.reduce_mean(self.loss_pieces_out[1])))
        self.other_summary_scalars.append(tf.summary.scalar("p_y", tf.reduce_mean(self.loss_pieces_out[2])))
        self.other_summary_scalars.append(tf.summary.scalar("v_xx", tf.reduce_mean(self.loss_pieces_out[3])))
        self.other_summary_scalars.append(tf.summary.scalar("v_yy", tf.reduce_mean(self.loss_pieces_out[4])))

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _get_optimizer(self):
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': self.max_optimizer_iterations,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
        return optimizer

    def get_max_optimizer_iterations(self):
        return self.max_optimizer_iterations

    def set_max_optimizer_iterations(self, max_iterations_in):
        self.max_optimizer_iterations = max_iterations_in

    def reset_training_data(self, x, y, t, r, u, v, bc_codes):
        X = np.concatenate([x, y, t, r], 1)

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
        self.r = X[:, 3:4]

        self.u = u
        self.v = v

        self.bc_codes = bc_codes


    # Initialize the class
    def __init__(self, x, y, t, r, u, v, bc_codes, layers, pressure_reference, discover_navier_stokes_parameters,
                 true_viscosity_in, true_density_in, max_optimizer_iterations_in):
        self.predicted_ref_node_pressures = 0.0
        self.discover_navier_stokes_parameters = discover_navier_stokes_parameters

        self.reset_training_data(x, y, t, r, u, v, bc_codes)

        self.ref_nodes_with_true_pressures = pressure_reference

        self.layers = layers

        self.iteration_counter = 0

        self.true_viscosity = true_viscosity_in
        self.true_density = true_density_in

        self.max_optimizer_iterations = max_optimizer_iterations_in
        self._finalise_state_setup()

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for layer in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[layer], layers[layer+1]])
            b = tf.Variable(tf.zeros([1, layers[layer+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS(self, x, y, t, r):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        # bias_term = tf.fill([tf.shape(x)[0], 1], 1.0)

        psi_and_p = self.neural_net(tf.concat([x, y, t, r], 1), self.weights, self.biases)

        psi = psi_and_p[:, 0]  # Seems to be that this is a scalar potential for the velocity, and that is what we predict
        p = psi_and_p[:, 1]  # Pressure field

        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]

        unsteady_flow = False
        # smooth_between_data = True

        if unsteady_flow:
            u_t = tf.gradients(u, t)[0]
        # if smooth_between_data:
        #     psi_t = tf.gradients(psi, t)[0]
        #     p_t = tf.gradients(p, t)[0]
        # else:
        #     psi_t = None
        #     p_t = None
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        if unsteady_flow:
            v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        if unsteady_flow:
            f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
            f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)
        else:
            f_u = lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
            f_v = lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)

        loss_pieces = [u[0]*v_x[0], v[0]*v_y[0], p_y[0], v_xx[0], v_yy[0]]

        # run the NN a second time on the first node (at x=1, y=-2, t=0) to get a reference pressure whose value we can
        # target in order to control the pressure field's absolute values;
        self.predicted_ref_node_pressures = []
        if self.ref_nodes_with_true_pressures is not None:
            for reference_node in self.ref_nodes_with_true_pressures:
                psi_and_p_ignore_psi = self.neural_net(tf.constant(reference_node[0:3], shape=(1, 4)), self.weights, self.biases)
                self.predicted_ref_node_pressures.append(psi_and_p_ignore_psi[:, 1])
        else:
            self.predicted_ref_node_pressures = [tf.constant([-1.0])]

        return u, v, p, f_u, f_v, psi, self.predicted_ref_node_pressures, loss_pieces  #, psi_t, p_t

    def callback(self, loss, lambda_1, lambda_2):
        self.iteration_counter += 1

        self.loss_history = array_extending_insert(self.loss_history, self.loss_history_write_index, loss)

        self.loss_history_write_index += 1

        print('(A) Loss: %.3e, l1: %.3f, l2: %.5f (%f/%f)' % (loss, lambda_1, lambda_2,
                                                              self.iteration_counter, self.max_optimizer_iterations))

    def train_model(self, nIter, summary_writer, x_predict, y_predict, t_predict, savefile_tag=None):

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t, self.r_tf: self.r,
                   self.u_tf: self.u, self.v_tf: self.v, self.bc_codes_tf: self.bc_codes}

        self.loss_history = np.zeros(self.max_optimizer_iterations + 1)  # 1 extra as a marker of where it switches between the optimizers
        self.loss_history_write_index = 0

        start_time = time.time()
        for it in tqdm.tqdm(range(nIter), desc='[Training (Param. set I.D. = {})...]'.format(savefile_tag)):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 40 == 0:
                elapsed = time.time() - start_time
                loss_value, loss_summary_retrieved, loss_vel_summary_retrieved, loss_ns_summary_retrieved,\
                other_summary_scalars_retrieved, loss_bc_summary_retrieved, loss_pressure_node_summary_retrieved = \
                                                    self.sess.run([self.loss, self.loss_summary,
                                                                  self.loss_velocity_summary, self.loss_ns_summary,
                                                                   self.other_summary_scalars, self.loss_bc_summary,
                                                                   self.loss_pressure_node_summary],
                                                                  tf_dict)
                self.loss_history = array_extending_insert(self.loss_history, self.loss_history_write_index, loss_value)
                self.loss_history_write_index += 1

                summary_writer.add_summary(loss_summary_retrieved, it)
                summary_writer.add_summary(loss_vel_summary_retrieved, it)
                summary_writer.add_summary(loss_ns_summary_retrieved, it)
                summary_writer.add_summary(loss_bc_summary_retrieved, it)
                summary_writer.add_summary(loss_pressure_node_summary_retrieved, it)
                for summary in other_summary_scalars_retrieved:
                    summary_writer.add_summary(summary, it)

                # if it % 5000 == 0:
                    # Show an image
                #TODO make this plot meshes - probably resample onto a regular grid to do this
                    # _, _, p_star, _ = self.predict(x_predict, y_predict, t_predict)
                    # image = np.reshape(p_star, (-1, 50, 100, 1))
                    # summary_image_op = tf.summary.image("Snapshot", image)
                    # summary_image = self.sess.run(summary_image_op)
                    # summary_writer.add_summary(summary_image)

                if self.discover_navier_stokes_parameters:
                    lambda_1_value = self.sess.run(self.lambda_1)
                    lambda_2_value = self.sess.run(self.lambda_2)
                else:
                    lambda_1_value = self.true_density
                    lambda_2_value = self.true_viscosity

                reference_node_pressure = self.sess.run(self.predicted_ref_node_pressures)
                print('It: %d, (B) Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f, %f' %
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed, reference_node_pressure[0]))
                start_time = time.time()

        # just a marker to see where the optimizers switched over
        self.loss_history = array_extending_insert(self.loss_history, self.loss_history_write_index, -1.0)

        self.loss_history_write_index += 1

        self._get_optimizer().minimize(self.sess,
                                       feed_dict = tf_dict,
                                       fetches = [self.loss, self.lambda_1, self.lambda_2],
                                       loss_callback = self.callback)

    def call_just_LBGDSF_optimizer(self):
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t, self.r_tf: self.r,
                   self.u_tf: self.u, self.v_tf: self.v, self.bc_codes_tf: self.bc_codes}

        self._get_optimizer().minimize(self.sess,
                                       feed_dict=tf_dict,
                                       fetches=[self.loss, self.lambda_1, self.lambda_2],
                                       loss_callback=self.callback)

    def predict(self, x_star, y_star, t_star, r_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star, self.r_tf: r_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        psi_pred = self.sess.run(self.psi_pred, tf_dict)

        return u_star, v_star, p_star, psi_pred

    def getLossHistory(self):
        return self.loss_history

    def get_loss(self, x_star, y_star, t_star, r_star, u_star, v_star, boundary_condition_codes):
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star, self.r_tf: r_star,
                   self.bc_codes_tf: boundary_condition_codes}

        navier_stokes_loss = self.sess.run(self.loss_navier_stokes, tf_dict)
        boundary_condition_loss = self.sess.run(self.loss_boundary_conditions, tf_dict)

        print('boundary_condition_loss', boundary_condition_loss)

        loss_dict = {'navier_stokes_loss': navier_stokes_loss,
                     'boundary_condition_loss': boundary_condition_loss}

        velocity_training_data_available = u_star is not None
        if velocity_training_data_available:
            tf_dict[self.u_tf] = u_star
            tf_dict[self.v_tf] = v_star

            total_loss = self.sess.run(self.loss, tf_dict)
            loss_dict['total_loss'] = total_loss

        return loss_dict

    def get_solution(self, x_star, y_star, t_star, r_star):
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star, self.r_tf: r_star}
        return self.sess.run(self.u_pred, tf_dict)


def plot_graph(x_data, y_data, title, scatter_x=None, scatter_y=None, savefile_nametag=None, second_y_data=None,
               y_range_1=(None, None), y_range_2=(None, None), y_range_3=(None, None),
               relative_or_absolute_output_folder=None,
               logplot=True):
    plt.figure(88)
    if second_y_data is not None:
        number_of_columns = 3
    else:
        number_of_columns = 1

    plt.subplot(1, number_of_columns, 1)
    plt.plot(x_data, y_data)
    plt.axis([None, None, y_range_1[0], y_range_1[1]])
    if logplot:
        plt.yscale('log')
    plt.title(title)

    if second_y_data is not None:
        plt.subplot(1, number_of_columns, 2)
        plt.plot(x_data, second_y_data)
        plt.axis([None, None, y_range_2[0], y_range_2[1]])
        if logplot:
            plt.yscale('log')
        plt.title('boundary')

        plt.subplot(1, number_of_columns, 3)
        loss_sum = [y1+y2 for (y1,y2) in zip(y_data, second_y_data)]
        plt.plot(x_data, loss_sum)
        plt.axis([None, None, y_range_3[0], y_range_3[1]])
        if logplot:
            plt.yscale('log')
        plt.title('sum')

    if scatter_x is not None and scatter_y is not None:
        plt.subplot(1, number_of_columns, 1)
        plt.scatter(scatter_x, scatter_y, c='red')
        if second_y_data is not None:
            plt.subplot(1, number_of_columns, 2)
            boundary_loss_interpolator = scipy.interpolate.interp1d(x_data, second_y_data)
            plt.scatter(scatter_x, boundary_loss_interpolator(scatter_x), c='red')
            plt.subplot(1, number_of_columns, 3)
            total_loss_interpolator = scipy.interpolate.interp1d(x_data, loss_sum)
            plt.scatter(scatter_x, total_loss_interpolator(scatter_x), c='red')

    if savefile_nametag is not None:
        title += savefile_nametag

    figure_savefile = title.replace(" ", "_") + '.png'
    if relative_or_absolute_output_folder is not None:
        if not os.path.exists(relative_or_absolute_output_folder):
            os.mkdir(relative_or_absolute_output_folder)
        figure_savefile = relative_or_absolute_output_folder + figure_savefile
    plt.savefig(figure_savefile)
    plt.close()


def plot_solution(X_star, u_star, title, colour_range=(None, None), relative_or_absolute_folder_path=None):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

    plt.figure(99)
    plt.pcolor(X, Y, U_star, cmap='jet', vmin=colour_range[0], vmax=colour_range[1])
    plt.colorbar()
    plt.title(title)
    figure_savefile = title.replace(" ", "_") + '.png'
    if relative_or_absolute_folder_path is not None:
        figure_savefile = relative_or_absolute_folder_path + figure_savefile
        if not os.path.exists(relative_or_absolute_folder_path):
            os.mkdir(relative_or_absolute_folder_path)
    plt.savefig(figure_savefile)
    plt.close()


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def train_and_pickle_model(tensorboard_log_directory_in, model_in, number_of_training_iterations_in, x_star_in, y_star_in,
                           t_star_in, pickled_model_filename_in, saved_tf_model_filename_in, savefile_tag_in):
    summary_writer = tf.summary.FileWriter(tensorboard_log_directory_in, tf.get_default_graph())

    model_in.train_model(number_of_training_iterations_in, summary_writer, x_star_in, y_star_in, t_star_in, savefile_tag_in)

    summary_writer.flush()
    summary_writer.close()

    # pickle
    try:
        with open(pickled_model_filename_in, 'wb') as pickled_model_file:
            pickle.dump(model_in, pickled_model_file)
        try:
            os.remove(saved_tf_model_filename_in + '.index')
        except FileNotFoundError:
            pass
        tf.train.Saver().save(model_in.sess, saved_tf_model_filename_in)
    except TypeError as e:
        error_message = "Error pickling model: model not saved!"
        print(error_message, e)
        log_message(error_message)


def _load_model(pickled_model_filename, saved_tf_model_filename, max_optimizer_iterations_in=None):
    tf.reset_default_graph()
    with open(pickled_model_filename, 'rb') as pickled_model_file:
        loaded_model = pickle.load(pickled_model_file)
        if max_optimizer_iterations_in:
            loaded_model.set_max_optimizer_iterations(max_optimizer_iterations_in)

    loaded_model.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                  log_device_placement=False))

    tf.train.Saver().restore(loaded_model.sess, saved_tf_model_filename)
    return loaded_model


def load_and_evaluate_model_at_point(point, pickled_model_filename, saved_tf_model_filename,
                                     max_optimizer_iterations_in):

    model = _load_model(pickled_model_filename, saved_tf_model_filename)
    prediction = evaluate_solution(model, point)

    return prediction


class BoundaryErrorModes(object):
    VS_ANALYTIC_BC = 1
    VS_FEM_SOLUTION = 2


def get_model_error_on_boundary(pickled_model_filename, saved_tf_model_filename,
                                max_optimizer_iterations_in, test_vtu_filename,
                                test_parameters_container,
                                master_model_data_root_path,
                                inflow_lambda,
                                mode=BoundaryErrorModes.VS_ANALYTIC_BC):

    vtk_reader = VtkDataReader.VtkDataReader(test_vtu_filename,
                                             test_parameters_container,
                                             master_model_data_root_path
                                             )

    model_test_input = vtk_reader.get_test_data(VtkDataReader.VtkDataReader.MODE_UNSTRUCTURED)

    model = _load_model(pickled_model_filename, saved_tf_model_filename)
    prediction = evaluate_solution(model, model_test_input)

    noslip_locations = np.where(prediction['bc_codes'] == BC.Codes.NOSLIP)

    errors = dict()

    if mode == BoundaryErrorModes.VS_ANALYTIC_BC:
        u_err_squared = np.sum(
            np.square(prediction['u_pred'][noslip_locations] - 0.0))

        v_err_squared = np.sum(
            np.square(prediction['v_pred'][noslip_locations] - 0.0))

    elif mode == BoundaryErrorModes.VS_FEM_SOLUTION:
        u_err_squared = np.sum(
            np.square(prediction['u_pred'][noslip_locations] - model_test_input['u'][noslip_locations]))

        v_err_squared = np.sum(
            np.square(prediction['v_pred'][noslip_locations] - model_test_input['v'][noslip_locations]))

    else:
        raise RuntimeError("Unknown mode set in get_model_error_on_boundary.")

    velocity_error = np.sqrt(u_err_squared + v_err_squared) / np.sum(noslip_locations)
    errors['noslip_velocity'] = velocity_error

    p_err_squared = np.sum(
        np.square(prediction['p_pred'][noslip_locations] - model_test_input['p'][noslip_locations])) / np.sum(
        noslip_locations)
    pressure_error = np.sqrt(p_err_squared)

    errors['noslip_pressure'] = pressure_error

    boundary_node_count = 0
    inflow_squared_velocity_error_accumulator = 0
    inflow_squared_pressure_error_accumulator = 0
    for index, node_code in enumerate(prediction['bc_codes']):
        if node_code == BC.Codes.INFLOW:
            boundary_node_count += 1

            if mode == BoundaryErrorModes.VS_ANALYTIC_BC:
                node_x = prediction['x'][index]
                node_y = prediction['y'][index]
                inflow_bc = inflow_lambda(node_x, node_y)

                inflow_squared_velocity_error_accumulator += np.square(prediction['u_pred'][index] - inflow_bc[0]) + np.square(
                    prediction['v_pred'][index] - inflow_bc[1])

            elif mode == BoundaryErrorModes.VS_FEM_SOLUTION:
                inflow_squared_velocity_error_accumulator += np.square(
                    prediction['u_pred'][index] - model_test_input['u'][index]) + np.square(
                    prediction['v_pred'][index] - model_test_input['v'][index])

                # inflow_squared_velocity_error_accumulator += np.square(
                #     prediction['p_pred'][index] - model_test_input['p'][index]) + np.square(
                #     prediction['v_pred'][index] - model_test_input['v'][index])
            else:
                raise RuntimeError("Unknown mode set in get_model_error_on_boundary.")

    inflow_velocity_error = np.sqrt(inflow_squared_velocity_error_accumulator) / boundary_node_count

    errors['inflow_velocity_error'] = inflow_velocity_error[0]

    return errors


def get_model_prediction(pickled_model_filename, saved_tf_model_filename, model_test_input):

    model = _load_model(pickled_model_filename, saved_tf_model_filename)
    prediction = evaluate_solution(model, model_test_input)

    return prediction


def get_test_data_from_vtu(test_vtu_filename, test_parameters_container, domain_only=False):
    config_manager = ConfigManager.ConfigManager()
    master_model_data_root_path = config_manager.get_master_model_data_root_path()

    vtk_reader = VtkDataReader.VtkDataReader(test_vtu_filename,
                                             test_parameters_container,
                                             master_model_data_root_path
                                             )

    test_data = vtk_reader.get_test_data(VtkDataReader.VtkDataReader.MODE_UNSTRUCTURED, domain_only=domain_only)
    return test_data


def save_prediction_to_vtu(test_vtu_filename, test_parameters_container, prediction, fem_array_names_to_add=('u', 'v', 'p')):
    config_manager = ConfigManager.ConfigManager()
    master_model_data_root_path = config_manager.get_master_model_data_root_path()

    vtk_reader = VtkDataReader.VtkDataReader(test_vtu_filename,
                                             test_parameters_container,
                                             master_model_data_root_path
                                             )

    vtk_reader.save_prediction_and_errors_to_vtk_mesh(prediction, fem_array_names_to_add=fem_array_names_to_add)


def load_and_evaluate_model_against_full_solution(pickled_model_filename, saved_tf_model_filename,
                                                  true_density_value, true_viscosity_value, test_vtu_filename,
                                                  test_parameters_container,
                                                  generate_vtu_if_needed=False,
                                                  domain_only=False):

    if generate_vtu_if_needed and not os.path.exists(test_vtu_filename):
        config_manager = ConfigManager.ConfigManager()
        data_path = config_manager.get_nektar_data_root_path()
        simulation_folder_template = config_manager.get_mesh_data_folder_template()
        vtu_and_xml_file_basename = config_manager.get_vtu_and_xml_file_basename()

        # generate vtu
        mesh_xml_file_name_template = data_path + simulation_folder_template + '/' + vtu_and_xml_file_basename + '.xml'
        mesh_xml_file_name = mesh_xml_file_name_template.format(test_parameters_container.get_t(), test_parameters_container.get_r())

        vtu_file_name_template = data_path + simulation_folder_template + '/' + vtu_and_xml_file_basename + '.vtu'
        vtu_file_name = vtu_file_name_template.format(test_parameters_container.get_t(), test_parameters_container.get_r())

        mesh_generator = Meshing.MeshGenerator(mesh_xml_file_name, test_parameters_container)
        mesh_generator.generate_in_correct_folder(vtu_additional_file_name=vtu_file_name)

        ActiveLearningUtilities.convert_xml_to_vtu(mesh_xml_file_name, test_vtu_filename)


    test_data = get_test_data_from_vtu(test_vtu_filename, test_parameters_container, domain_only=domain_only)

    prediction = get_model_prediction(pickled_model_filename, saved_tf_model_filename, test_data)

    errors_out = None

    if domain_only:
        save_prediction_to_vtu(test_vtu_filename, test_parameters_container, prediction, fem_array_names_to_add=None)
    else:
        save_prediction_to_vtu(test_vtu_filename, test_parameters_container, prediction)
        compute_errors(test_data, prediction, true_density_value, true_viscosity_value)
        print("errors_out:", errors_out)

    return errors_out


def compute_errors(test_data, prediction, true_density_value, true_viscosity_value):
    u_test = test_data['u']
    v_test = test_data['v']
    p_test = test_data['p']

    # Error
    errors = dict()
    errors['error_u'] = np.linalg.norm(u_test - prediction['u_pred'], 2) / np.linalg.norm(u_test, 2)
    errors['error_v'] = np.linalg.norm(v_test - prediction['v_pred'], 2) / np.linalg.norm(v_test, 2)
    errors['error_p'] = np.linalg.norm(np.squeeze(p_test) - prediction['p_pred'], 2) / np.linalg.norm(p_test, 2)
    log_message('Computed errors were: {}'.format(errors))

    errors['error_lambda_1'] = np.abs(prediction['lambda_1_value'] - true_density_value) / true_density_value * 100
    errors['error_lambda_2'] = np.abs(prediction['lambda_2_value'] - true_viscosity_value) / true_viscosity_value * 100

    # print('Error u: %e' % (error_u))
    # print('Error v: %e' % (error_v))
    # print('Error p: %e' % (error_p))
    # print('Error l1: %.5f%%' % (error_lambda_1))
    # print('Error l2: %.5f%%' % (error_lambda_2))
    # print(lambda_1_value, lambda_2_value)

    # Plot Results
    # plot_solution(X_star, u_pred, "Predicted Velocity U")
    # plot_solution(X_star, v_pred, "Predicted Velocity V")
    # plot_solution(X_star, p_pred, "Predicted Pressure")
    # plot_solution(X_star, p_test, "True Pressure")
    # plot_solution(X_star, p_test[:, 0] - p_pred, "Pressure Error")
    # plot_solution(X_star, psi_pred, "Psi")

    return errors


def evaluate_solution(model, test_data):

    prediction = dict()
    u_pred, v_pred, p_pred, psi_pred = model.predict(test_data['x'], test_data['y'], test_data['t'], test_data['r'])

    prediction['u_pred'] = u_pred
    prediction['v_pred'] = v_pred
    prediction['p_pred'] = p_pred
    prediction['psi_pred'] = psi_pred

    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)

    prediction['lambda_1_value'] = lambda_1_value
    prediction['lambda_2_value'] = lambda_2_value

    # Add the test coordinates, as useful metadata about the prediction
    prediction['x'] = test_data['x']
    prediction['y'] = test_data['y']
    prediction['t'] = test_data['t']
    prediction['r'] = test_data['r']
    try:
        prediction['bc_codes'] = test_data['bc_codes']
    except KeyError:
        warnings.warn('Key bc_codes was not present in this test_data.')

    return prediction


def run_NS_trainer(input_pickle_file_template, input_saved_model_template, savefile_tag, number_of_training_iterations,
                   use_pressure_reference_in_training, max_optimizer_iterations_in,
                   N_train_specifier,
                   true_viscosity_value,
                   true_density_value,
                   discover_navier_stokes_parameters = False,
                   load_existing_model=False, additional_simulation_data=None, parent_logger=None,
                   data_caching_directory=os.getcwd()):

    tensorboard_log_directory_base = '{}/logs'.format(data_caching_directory)

    if not os.path.exists(tensorboard_log_directory_base):
        os.mkdir(tensorboard_log_directory_base)

    # Warning: this assumes that the only contents of the logs directory is subdirs with integer names.
    integer_log_subdir_names = [int(filename) for filename in os.listdir(tensorboard_log_directory_base)]
    try:
        next_available_integer_for_subdir_name = max(integer_log_subdir_names) + 1
    except ValueError:
        next_available_integer_for_subdir_name = 0

    tensorboard_log_directory = '{}/{}'.format(tensorboard_log_directory_base, next_available_integer_for_subdir_name)
    os.mkdir(tensorboard_log_directory)

    # Remove any files from the current directory, which contain "tempstate" - these can break future simulations if
    # left in-place.
    #
    # List required to trigger evaluation of the generator returned by map
    list(map(lambda filename: os.remove(filename),
            filter(lambda filename: 'tempstate' in filename, os.listdir('.'))
             ))

    with tf.device("/gpu:0"), TensorboardTools.TensorboardProcess(tensorboard_log_directory_base, 6006):

        timer = ttictoc.TicToc()
        timer.tic()

        do_noisy_data_case = False
        plot_lots = False
        train_model_further_with_new_data = True
        # use_pressure_reference_in_training = True
        # number_of_training_iterations = 100000  # 200000

        config_manager = ConfigManager.ConfigManager()
        layers = [4] + [config_manager.get_network_width()] * config_manager.get_number_of_hidden_layers() + [3]
        # layers = [3, 20, 20, 20, 20, 20, 2]

        # Load Data
        # training_data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')

        # data_directory = r'/home/chris/WorkData/nektar++/actual/tube_10mm_diameter_pt2Mesh_correctViscosity/'
        # training_data = VtkDataReader.VtkDataReader.from_single_data_file(data_directory + vtu_data_file_name + '.vtu'
        #                                            ).get_pinns_format_input_data()

        data_reader = VtkDataReader.MultipleFileReader(data_caching_directory, mode='unstructured')
        # base_base_data_directory = r'/home/chris/WorkData/nektar++/actual/'
        # file_names_and_parameter_values = [(data_directory + vtu_data_file_name, 1.0),
        #                                    (base_base_data_directory + r'tube_10mm_diameter_pt2Mesh_correctViscosity_doubleInflow/' + vtu_data_file_name, 2.0),
        #                                    (base_base_data_directory + r'tube_10mm_diameter_pt2Mesh_correctViscosity_1pt5Inflow/' + vtu_data_file_name, 1.5),
        #                                    (base_base_data_directory + r'tube_10mm_diameter_pt2Mesh_correctViscosity_2pt5Inflow/' + vtu_data_file_name, 2.5),
        #                                    (base_base_data_directory + r'tube_10mm_diameter_pt2Mesh_correctViscosity_3pt0Inflow/' + vtu_data_file_name, 3.0),
        #                                    (base_base_data_directory + r'tube_10mm_diameter_pt2Mesh_correctViscosity_4pt0Inflow/' + vtu_data_file_name, 4.0),
        #                                    (base_base_data_directory + r'tube_10mm_diameter_pt2Mesh_correctViscosity_3pt5Inflow/' + vtu_data_file_name, 3.5)] + \
        #                                     additional_simulation_data
        file_names_and_parameter_values = additional_simulation_data
        for fn_and_pv in file_names_and_parameter_values:
            data_reader.add_file_name("{}_using_points_from_xml.vtu".format(fn_and_pv[0]), fn_and_pv[1])
        if parent_logger is not None:
            parent_logger.info("NavierStokes.py is loading the following datafiles: {}".format(file_names_and_parameter_values))

        # additional_navier_stokes_only_datapoints = [1.7, 2.3, 2.7, 5.0, 0.0]
        # for additional_ns_only_point in additional_navier_stokes_only_datapoints:
        #     data_reader.add_point_for_navier_stokes_loss_only(additional_ns_only_point)

        # For now, we recycle the T dimension as the generic parameter dimension for steady flow training_data. Can
        # adjust this later.
        training_data = data_reader.get_training_data()

        x = training_data['x']
        y = training_data['y']
        t = training_data['t']
        r = training_data['r']
        u = training_data['u']
        v = training_data['v']
        p = training_data['p']
        bc_codes = training_data['bc_codes']

        total_data_length = len(u)
        N_train_specifier.set_num_total_available_datapoints(total_data_length)
        N_train = N_train_specifier.get_number_to_use_in_training()

        parent_logger.info("Using {}% of the data for training ({}/{} points).".format(N_train/total_data_length*100,
                                                                                       N_train, total_data_length))

        ######################################################################
        ######################## Noiseles Data ###############################
        ######################################################################
        # Training Data
        idx = np.random.choice(x.shape[0], N_train, replace=False)
        x_train = x[idx, :]
        y_train = y[idx, :]
        t_train = t[idx, :]
        r_train = r[idx, :]
        u_train = u[idx, :]
        v_train = v[idx, :]
        p_train = p[idx, :]
        bc_codes_train = bc_codes[idx]

        # Test Data
        # TODO make this actually test on a slice of our choosing, rather than on a fuller slice of the training data. This is ok, but we can do better.
        test_data_parameters = SPM.SimulationParameterContainer(t_train[0, 0], r_train[0, 0])
        test_data = data_reader.get_test_data(test_data_parameters)

        X_star = test_data['X_star']  # TODO this is somewhat redundant; X_star and (X_test and Y_test) contain the same data. Refactor.
        X_test = test_data['x']
        Y_test = test_data['y']
        t_test = test_data['t']
        r_test = test_data['r']

        # Left to its own devices, the neural network will find a pressure field with the correct gradient, but not with
        # the correct absolute value. We can solve this by providing a reference point for each solution field.
        #
        # This is not necessary (you could just shift all pressure fields manually, after training), but it is tidier
        # to do it this way.
        pressure_reference_nodes = []
        if use_pressure_reference_in_training:
            pressure_reference_location = (20.0, 5.0)
            for input_data_parameters, _ in data_reader.get_test_data_over_all_known_files_generator():

                p_reference = data_reader.evaluate_field_at_point(pressure_reference_location[0],
                                                                  pressure_reference_location[1],
                                                                  'p',
                                                                  input_data_parameters)

                pressure_reference_node = [pressure_reference_location[0],
                                           pressure_reference_location[1],
                                           input_data_parameters.get_t(),
                                           input_data_parameters.get_r(),
                                           p_reference]
            pressure_reference_nodes.append(pressure_reference_node)
        else:
            pressure_reference_nodes = None

        pickled_model_filename = input_pickle_file_template.format(savefile_tag)
        saved_tf_model_filename = input_saved_model_template.format(savefile_tag)
        pickled_model_filename_out = input_pickle_file_template.format(savefile_tag + 1)
        saved_tf_model_filename_out = input_saved_model_template.format(savefile_tag + 1)

        if load_existing_model:
            model = _load_model(pickled_model_filename, saved_tf_model_filename, max_optimizer_iterations_in)

            # model.call_just_LBGDSF_optimizer()
            if train_model_further_with_new_data:
                model.reset_training_data(x_train, y_train, t_train, r_train, u_train, v_train, bc_codes_train)

                train_and_pickle_model(tensorboard_log_directory, model, number_of_training_iterations, X_test, Y_test,
                                       t_test, pickled_model_filename_out, saved_tf_model_filename_out, savefile_tag)
        else:
            # Training
            model = PhysicsInformedNN(x_train, y_train, t_train, r_train, u_train, v_train, bc_codes_train, layers,
                                      pressure_reference_nodes,
                                      discover_navier_stokes_parameters, true_viscosity_value, true_density_value,
                                      max_optimizer_iterations_in)

            train_and_pickle_model(tensorboard_log_directory, model, number_of_training_iterations, X_test, Y_test,
                                   t_test, pickled_model_filename_out, saved_tf_model_filename_out, savefile_tag)

        gathered_errors_u = dict()
        gathered_errors_v = dict()
        gathered_errors_p = dict()

        for parameter_container, evaluation_test_data in data_reader.get_test_data_over_all_known_files_generator():
            prediction = evaluate_solution(model, evaluation_test_data)
            computed_errors = compute_errors(evaluation_test_data, prediction, true_density_value, true_viscosity_value)

            gathered_errors_u[parameter_container] = computed_errors['error_u']
            gathered_errors_v[parameter_container] = computed_errors['error_v']
            gathered_errors_p[parameter_container] = computed_errors['error_p']

        gathered_errors_u_list = []
        for index, key in enumerate(gathered_errors_u):
            gathered_errors_u_list.append(abs(gathered_errors_u[key]))
            parent_logger.info('gathered errors u axis label {} = {}'.format(index, key))

        gathered_errors_v_list = []
        for key in gathered_errors_v:
            gathered_errors_v_list.append(abs(gathered_errors_v[key]))

        gathered_errors_p_list = []
        for key in gathered_errors_p:
            gathered_errors_p_list.append(abs(gathered_errors_p[key]))

        all_gathered_errors = [gathered_errors_u_list, gathered_errors_v_list, gathered_errors_p_list]
        for gathered_errors in all_gathered_errors:
            for index, error in enumerate(gathered_errors):
                if not np.isfinite(error):
                    gathered_errors[index] = 0.0000012345



        plot_graph(range(len(gathered_errors_u_list)), gathered_errors_u_list, 'gathered errors u list', logplot=False)
        plot_graph(range(len(gathered_errors_v_list)), gathered_errors_v_list, 'gathered errors v list', logplot=False)
        plot_graph(range(len(gathered_errors_p_list)), gathered_errors_p_list, 'gathered errors p list', logplot=False)

        # np.savetxt('loss_history_{}_{}.dat'.format(number_of_training_iterations, model.get_max_optimizer_iterations()), model.getLossHistory())
        #
        # np.savetxt("p_pred_saved.dat", p_pred)
        # np.savetxt("X_star_saved.dat", X_star)

        # print("Pressure at 2nd node: True: {}, Predicted: {}, Difference: {}". format(p_test[0], p_pred[0], p_test[0]-p_pred[0]))

        # Predict for plotting
        lb = X_star.min(0)
        ub = X_star.max(0)
        nn = 200
        x = np.linspace(lb[0], ub[0], nn)
        y = np.linspace(lb[1], ub[1], nn)
        X, Y = np.meshgrid(x,y)

        # UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
        # VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
        # PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
        # P_exact = griddata(X_star, p_test.flatten(), (X, Y), method='cubic')

        if do_noisy_data_case:
            ######################################################################
            ########################### Noisy Data ###############################
            ######################################################################
            noise = 0.01
            u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
            v_train = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])

            # Training
            # model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
            model.train_model(number_of_training_iterations)  # 200000

            lambda_1_value_noisy = model.sess.run(model.lambda_1)
            lambda_2_value_noisy = model.sess.run(model.lambda_2)

            error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)*100
            error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.01)/0.01 * 100

            print('Error l1: %.5f%%' % (error_lambda_1_noisy))
            print('Error l2: %.5f%%' % (error_lambda_2_noisy))



        # ######################################################################
        # ############################# Plotting ###############################
        # ######################################################################
        #  # Load Data
        # data_vort = scipy.io.loadmat('../Data/cylinder_nektar_t0_vorticity.mat')
        #
        # x_vort = data_vort['x']
        # y_vort = data_vort['y']
        # w_vort = data_vort['w']
        # modes = data_vort['modes'].item()
        # nel = data_vort['nel'].item()
        #
        # xx_vort = np.reshape(x_vort, (modes+1,modes+1,nel), order = 'F')
        # yy_vort = np.reshape(y_vort, (modes+1,modes+1,nel), order = 'F')
        # ww_vort = np.reshape(w_vort, (modes+1,modes+1,nel), order = 'F')
        #
        # box_lb = np.array([1.0, -2.0])
        # box_ub = np.array([8.0, 2.0])
        #
        # fig, ax = newfig(1.0, 1.2)
        # ax.axis('off')
        #
        # ####### Row 0: Vorticity ##################
        # gs0 = gridspec.GridSpec(1, 2)
        # gs0.update(top=1-0.06, bottom=1-2/4 + 0.12, left=0.0, right=1.0, wspace=0)
        # ax = plt.subplot(gs0[:, :])
        #
        # for i in range(0, nel):
        #     h = ax.pcolormesh(xx_vort[:,:,i], yy_vort[:,:,i], ww_vort[:,:,i], cmap='seismic',shading='gouraud',  vmin=-3, vmax=3)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(h, cax=cax)
        #
        # ax.plot([box_lb[0],box_lb[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
        # ax.plot([box_ub[0],box_ub[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
        # ax.plot([box_lb[0],box_ub[0]],[box_lb[1],box_lb[1]],'k',linewidth = 1)
        # ax.plot([box_lb[0],box_ub[0]],[box_ub[1],box_ub[1]],'k',linewidth = 1)
        #
        # ax.set_aspect('equal', 'box')
        # ax.set_xlabel('$x$')
        # ax.set_ylabel('$y$')
        # ax.set_title('Vorticity', fontsize = 10)
        #
        #
        # ####### Row 1: Training training_data ##################
        # ########      u(t,x,y)     ###################
        # gs1 = gridspec.GridSpec(1, 2)
        # gs1.update(top=1-2/4, bottom=0.0, left=0.01, right=0.99, wspace=0)
        # ax = plt.subplot(gs1[:, 0],  projection='3d')
        # ax.axis('off')
        #
        # r1 = [x_star.min(), x_star.max()]
        # r2 = [training_data['t'].min(), training_data['t'].max()]
        # r3 = [y_star.min(), y_star.max()]
        #
        # for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
        #     if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
        #         ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)
        #
        # ax.scatter(x_train, t_train, y_train, s = 0.1)
        # ax.contourf(X,UU_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)
        #
        # ax.text(x_star.mean(), training_data['t'].min() - 1, y_star.min() - 1, '$x$')
        # ax.text(x_star.max()+1, training_data['t'].mean(), y_star.min() - 1, '$t$')
        # ax.text(x_star.min()-1, training_data['t'].min() - 0.5, y_star.mean(), '$y$')
        # ax.text(x_star.min()-3, training_data['t'].mean(), y_star.max() + 1, '$u(t,x,y)$')
        # ax.set_xlim3d(r1)
        # ax.set_ylim3d(r2)
        # ax.set_zlim3d(r3)
        # axisEqual3D(ax)
        #
        # ########      v(t,x,y)     ###################
        # ax = plt.subplot(gs1[:, 1],  projection='3d')
        # ax.axis('off')
        #
        # r1 = [x_star.min(), x_star.max()]
        # r2 = [training_data['t'].min(), training_data['t'].max()]
        # r3 = [y_star.min(), y_star.max()]
        #
        # for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
        #     if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
        #         ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)
        #
        # ax.scatter(x_train, t_train, y_train, s = 0.1)
        # ax.contourf(X,VV_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)
        #
        # ax.text(x_star.mean(), training_data['t'].min() - 1, y_star.min() - 1, '$x$')
        # ax.text(x_star.max()+1, training_data['t'].mean(), y_star.min() - 1, '$t$')
        # ax.text(x_star.min()-1, training_data['t'].min() - 0.5, y_star.mean(), '$y$')
        # ax.text(x_star.min()-3, training_data['t'].mean(), y_star.max() + 1, '$v(t,x,y)$')
        # ax.set_xlim3d(r1)
        # ax.set_ylim3d(r2)
        # ax.set_zlim3d(r3)
        # axisEqual3D(ax)
        #
        # # savefig('./figures/NavierStokes_data')
        #
        #
        # fig, ax = newfig(1.015, 0.8)
        # ax.axis('off')
        #
        # ######## Row 2: Pressure #######################
        # ########      Predicted p(t,x,y)     ###########
        # gs2 = gridspec.GridSpec(1, 2)
        # gs2.update(top=1, bottom=1-1/2, left=0.1, right=0.9, wspace=0.5)
        # ax = plt.subplot(gs2[:, 0])
        # h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow',
        #               extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
        #               origin='lower', aspect='auto')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        #
        # fig.colorbar(h, cax=cax)
        # ax.set_xlabel('$x$')
        # ax.set_ylabel('$y$')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('Predicted pressure', fontsize = 10)
        #
        # ########     Exact p(t,x,y)     ###########
        # ax = plt.subplot(gs2[:, 1])
        # h = ax.imshow(P_exact, interpolation='nearest', cmap='rainbow',
        #               extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
        #               origin='lower', aspect='auto')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        #
        # fig.colorbar(h, cax=cax)
        # ax.set_xlabel('$x$')
        # ax.set_ylabel('$y$')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('Exact pressure', fontsize = 10)
        #
        #
        # ######## Row 3: Table #######################
        # gs3 = gridspec.GridSpec(1, 2)
        # gs3.update(top=1-1/2, bottom=0.0, left=0.0, right=1.0, wspace=0)
        # ax = plt.subplot(gs3[:, :])
        # ax.axis('off')
        #
        # s = r'$\begin{tabular}{|c|c|}';
        # s = s + r' \hline'
        # s = s + r' Correct PDE & $\begin{array}{c}'
        # s = s + r' u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})\\'
        # s = s + r' v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})'
        # s = s + r' \end{array}$ \\ '
        # s = s + r' \hline'
        # s = s + r' Identified PDE (clean training_data) & $\begin{array}{c}'
        # s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value, lambda_2_value)
        # s = s + r' \\'
        # s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value, lambda_2_value)
        # s = s + r' \end{array}$ \\ '
        # s = s + r' \hline'
        # s = s + r' Identified PDE (1\% noise) & $\begin{array}{c}'
        # if do_noisy_data_case:
        #     s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
        #     s = s + r' \\'
        #     s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
        # s = s + r' \end{array}$ \\ '
        # s = s + r' \hline'
        # s = s + r' \end{tabular}$'
        #
        # ax.text(0.015,0.0,s)

        timer.toc()
        print("Time taken to run: {}".format(timer.elapsed))

        # savefig('./figures/NavierStokes_prediction')


def evaluate_all_boundary_errors():
    for model_index_to_load in tqdm.tqdm(range(1, 62, 5)):
        max_optimizer_iterations = 50000
        data_root = '/home/chris/WorkData/nektar++/actual/bezier/master_data/'
        saved_tf_model_filename = os.path.join(data_root, 'saved_model_{}.tf'.format(model_index_to_load))
        pickled_model_filename = os.path.join(data_root, 'saved_model_{}.pickle'.format(model_index_to_load))

        config_manager = ConfigManager.ConfigManager()
        nektar_data_root_path = config_manager.get_nektar_data_root_path()
        simulation_subfolder_template = config_manager.get_mesh_data_folder_template()
        vtu_and_xml_file_basename = config_manager.get_vtu_and_xml_file_basename()
        master_model_data_root_path = config_manager.get_master_model_data_root_path()

        test_vtu_filename_template_without_extension = (nektar_data_root_path + simulation_subfolder_template +
                                                        vtu_and_xml_file_basename + r'_using_points_from_xml')

        parameter_descriptor_t = {'range_start': -2.0, 'range_end': 2.0}
        number_of_parameter_points_t = int(
            (parameter_descriptor_t['range_end'] - parameter_descriptor_t['range_start']) * 3) + 1
        parameter_descriptor_t['number_of_points'] = number_of_parameter_points_t

        parameter_descriptor_r = {'range_start': -2.0, 'range_end': 2.0}
        number_of_parameter_points_r = int(
            (parameter_descriptor_r['range_end'] - parameter_descriptor_r['range_start']) * 3) + 1
        parameter_descriptor_r['number_of_points'] = number_of_parameter_points_r

        simulation_parameter_manager = SPM.SimulationParameterManager(parameter_descriptor_t, parameter_descriptor_r)

        gathered_errors = dict()

        for parameters_container in tqdm.tqdm(simulation_parameter_manager.all_parameter_points(),
                                              total=simulation_parameter_manager.get_num_parameter_points()):

            test_vtu_filename_without_extension = test_vtu_filename_template_without_extension.format(
                parameters_container.get_t(), parameters_container.get_r())

            test_vtu_filename = test_vtu_filename_without_extension + '.vtu'

            inflow_lambda = lambda x, y: ((10.0 - y) * y / 25.0 * parameters_container.get_t(), 0.0)

            # fem_err = get_model_error_on_boundary(pickled_model_filename, saved_tf_model_filename,
            #                                   max_optimizer_iterations, test_vtu_filename,
            #                                   parameters_container,
            #                                   master_model_data_root_path,
            #                                   inflow_lambda,
            #                                   mode=BoundaryErrorModes.VS_FEM_SOLUTION)
            #
            # print('vs. FEM error:', fem_err)

            analytic_err = get_model_error_on_boundary(pickled_model_filename, saved_tf_model_filename,
                                                       max_optimizer_iterations, test_vtu_filename,
                                                       parameters_container,
                                                       master_model_data_root_path,
                                                       inflow_lambda,
                                                       mode=BoundaryErrorModes.VS_ANALYTIC_BC)

            print('vs. analytic error:', parameters_container, analytic_err)
            gathered_errors[parameters_container] = analytic_err

        with open(config_manager.get_boundary_errors_filename_template().format(model_index_to_load), 'wb') as outfile:
            pickle.dump(gathered_errors, outfile)


def read_boundary_errors(filename_template, model_index_to_load):
    with open(filename_template.format(model_index_to_load), 'rb') as infile:
        boundary_errors = pickle.load(infile)

    return boundary_errors


def predict_and_save_to_vtu_file(parameters_container, step_index):
    config_manager = ConfigManager.ConfigManager()

    pickled_model_filename = config_manager.get_master_model_data_root_path() + r'saved_model_{}.pickle'.format(step_index)
    saved_tf_model_filename = config_manager.get_master_model_data_root_path() + r'saved_model_{}.tf'.format(step_index)

    true_viscosity = 0.004
    true_density = 0.00106

    template = config_manager.get_nektar_data_root_path() + config_manager.get_mesh_data_folder_template() + \
               r'/tube_bezier_1pt0mesh.vtu'
    test_vtu_filename = template.format(parameters_container.get_t(), parameters_container.get_r())


    load_and_evaluate_model_against_full_solution(pickled_model_filename,
                                                  saved_tf_model_filename,
                                                  true_density, true_viscosity, test_vtu_filename,
                                                  parameters_container, generate_vtu_if_needed=True,
                                                  domain_only=True)


if __name__ == "__main__":
    # sim_dir_and_parameter_tuple = (r'/home/chris/WorkData/nektar++/actual/bezier/basic_t0.0/tube_bezier_1pt0mesh', 0.0)
    # additional_nametag = 'working_500TrainingDatapoints'
    # num_training_iterations = 100000
    # max_optimizer_iterations = 50000  # 50000
    # use_pressure_reference_in_training = True
    # vtu_data_file_name = 'tube_bezier_1pt0mesh'
    # savefile_tag = 4
    # number_of_hidden_layers = config_manager.get_number_of_hidden_layers()
    # true_viscosity = 0.004
    # true_density = 0.00106
    #
    # if use_pressure_reference_in_training:
    #     file_name_tag = vtu_data_file_name + additional_nametag + "_zero_ref_pressure.pickle"
    # else:
    #     file_name_tag = vtu_data_file_name + additional_nametag + ""
    #
    # file_name_tag = '{}_{}_layers'.format(file_name_tag, number_of_hidden_layers+2)
    #
    # input_pickle_file_template = 'retrained{{}}_retrained3_retrained2_retrained_trained_model_nonoise_{}{}.pickle'.format(
    #     num_training_iterations, file_name_tag)
    # input_saved_model_template = 'retrained{{}}_retrained3_retrained2_retrained_trained_model_nonoise_{}{}.tf'.format(
    #     num_training_iterations, file_name_tag)
    #
    # N_train_in = 5000
    #
    # run_NS_trainer(input_pickle_file_template, input_saved_model_template, savefile_tag, num_training_iterations,
    #                use_pressure_reference_in_training, max_optimizer_iterations,
    #                N_train_in, true_viscosity, true_density,
    #                load_existing_model=False, additional_simulation_data=[sim_dir_and_parameter_tuple])


    # evaluate_all_boundary_errors()


    param_container = SPM.SimulationParameterContainer(-2.0, 2.0)
    step_index = 31
    predict_and_save_to_vtu_file(param_container, step_index)
