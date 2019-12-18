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
from plotting import newfig, savefig  # this is the thing in the Utilities subfolder (added above to the sys path) - NOT a 3rd party module
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

        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred, self.psi_pred, p_at_first_node =\
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
        
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]

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
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
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
        
    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        # bias_term = tf.fill([tf.shape(x)[0], 1], 1.0)

        psi_and_p = self.neural_net(tf.concat([x, y, t], 1), self.weights, self.biases)

        psi = psi_and_p[:, 0]  # Seems to be that this is a scalar potential for the velocity, and that is what we predict
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

        f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
        f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)

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

        self.loss_history = np.zeros(self.max_optimizer_iterations + 1)  # 1 extra as a marker of where it switches between the optimizers
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
                    _, _, p_star = self.predict(x_predict, y_predict, t_predict)
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
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.lambda_1, self.lambda_2],
                                loss_callback = self.callback)

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
    X, Y = np.meshgrid(x,y)
    
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    
    plt.figure(index)
    plt.pcolor(X,Y,U_star, cmap = 'jet')
    plt.colorbar()
    plt.title(title)
    plt.savefig(title.replace(" ", "_") + '.png')
    
    
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


if __name__ == "__main__":

    tensorboard_log_directory = r'.\logs'

    # List required to trigger evaluation of the generator returned by map
    list(map(lambda filename: os.remove(tensorboard_log_directory + '/' + filename),
        os.listdir(tensorboard_log_directory)))


    # Remove any files from the current directory, which contain "tempstate" - these can break future simulations if
    # left in-place.
    list(map(lambda filename: os.remove(filename),
            filter(lambda filename: 'tempstate' in filename, os.listdir('.'))
             ))

    with tf.device("/gpu:0"), TensorboardTools.TensorboardProcess(tensorboard_log_directory, 6006):

        timer = ttictoc.TicToc()
        timer.tic()

        do_noisy_data_case = False
        load_existing_model = False
        use_pressure_node_in_training = False
        discover_navier_stokes_parameters = True
        number_of_training_iterations = 200000  # 200000

        N_train = 5000

        layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
        # layers = [3, 20, 20, 20, 20, 20, 2]

        # Load Data
        data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')

        U_star = data['U_star']  # N x 2 x T
        P_star = data['p_star']  # N x T
        t_star = data['t']  # T x 1
        X_star = data['X_star']  # N x 2

        N = X_star.shape[0]
        T = t_star.shape[0]

        # Rearrange Data - NOTE: the .T here is the transpose of the numpy array - it's not the same as the variable T in
        # the enclosing namespace
        XX = np.tile(X_star[:, 0], (1, T))  # N x T
        YY = np.tile(X_star[:, 1], (1, T))  # N x T
        TT = np.tile(t_star, (1, N)).T  # N x T

        UU = U_star[:,0,:]  # N x T
        VV = U_star[:,1,:]  # N x T
        PP = P_star  # N x T

        x = XX.flatten()[:,None]  # NT x 1
        y = YY.flatten()[:,None]  # NT x 1
        t = TT.flatten()[:,None]  # NT x 1

        u = UU.flatten()[:,None]  # NT x 1
        v = VV.flatten()[:,None]  # NT x 1
        p = PP.flatten()[:,None]  # NT x 1

        ######################################################################
        ######################## Noiseles Data ###############################
        ######################################################################
        # Training Data
        idx = np.random.choice(N*T, N_train, replace=False)
        x_train = x[idx, :]
        y_train = y[idx, :]
        t_train = t[idx, :]
        u_train = u[idx, :]
        v_train = v[idx, :]

        if use_pressure_node_in_training:
            file_name_tag = "_zero_ref_pressure.pickle"
            p_single_reference_node = 0.0  # p[995000+1] # on the last time slice
            loss_history_file_nametag = "single_reference_pressure"
        else:
            file_name_tag = ""
            p_single_reference_node = None
            loss_history_file_nametag = "no_reference_pressure"

        # file_name_tag = "_no_scipy_optimiser"

        # Test Data
        snap = np.array([100])
        x_star = X_star[:, 0:1]
        y_star = X_star[:, 1:2]
        t_star = TT[:, snap]

        file_name_tag = '{}_{}_layers'.format(file_name_tag, len(layers))

        pickled_model_filename = 'trained_model_nonoise_{}{}.pickle'.format(number_of_training_iterations, file_name_tag)
        saved_tf_model_filename = 'trained_model_nonoise_{}{}.tf'.format(number_of_training_iterations, file_name_tag)

        if load_existing_model:
            tf.reset_default_graph()
            with open(pickled_model_filename, 'rb') as pickled_model_file:
                model = pickle.load(pickled_model_file)

            model.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                         log_device_placement=True))

            tf.train.Saver().restore(model.sess, saved_tf_model_filename)
        else:
            # Training
            model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers, p_single_reference_node,
                                      discover_navier_stokes_parameters)

            summary_writer = tf.summary.FileWriter(tensorboard_log_directory, tf.get_default_graph())

            model.train(number_of_training_iterations, summary_writer, x_star, y_star, t_star)  # 200000

            summary_writer.flush()
            summary_writer.close()

            # pickle
            try:
                with open(pickled_model_filename, 'wb') as pickled_model_file:
                    pickle.dump(model, pickled_model_file)
                tf.train.Saver().save(model.sess, saved_tf_model_filename)
            except TypeError as e:
                print("Error pickling model: model not saved!", e)

        # Test Data
        snap = np.array([100])
        x_star = X_star[:,0:1]
        y_star = X_star[:,1:2]
        t_star = TT[:,snap]

        u_star = U_star[:,0,snap]
        v_star = U_star[:,1,snap]
        p_star = P_star[:,snap]

        # Prediction
        u_pred, v_pred, p_pred, psi_pred = model.predict(x_star, y_star, t_star)
        lambda_1_value = model.sess.run(model.lambda_1)
        lambda_2_value = model.sess.run(model.lambda_2)

        # Error
        error_u = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
        error_v = np.linalg.norm(v_star-v_pred, 2)/np.linalg.norm(v_star, 2)
        error_p = np.linalg.norm(p_star-p_pred, 2)/np.linalg.norm(p_star, 2)

        error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
        error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100

        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error p: %e' % (error_p))
        print('Error l1: %.5f%%' % (error_lambda_1))
        print('Error l2: %.5f%%' % (error_lambda_2))

        # Plot Results
        plot_solution(X_star, u_pred, 1, "Predicted Velocity U")
        plot_solution(X_star, v_pred, 2, "Predicted Velocity V")
        plot_solution(X_star, p_pred, 3, "Predicted Pressure")
        plot_solution(X_star, p_star, 4, "True Pressure")
        print("shapes: {} {}".format(p_star.shape, p_pred.shape))
        plot_solution(X_star, p_star[:, 0] - p_pred, 5, "Pressure Error")
        plot_solution(X_star, psi_pred, 6, "Psi")

        np.savetxt('loss_history_{}_{}_{}.dat'.format(number_of_training_iterations, model.getMaxOptimizerIterations(),
                                                      loss_history_file_nametag), model.getLossHistory())

        np.savetxt("p_pred_saved.dat", p_pred)
        np.savetxt("X_star_saved.dat", X_star)

        print("Pressure at 2nd node: True: {}, Predicted: {}, Difference: {}". format(p_star[1], p_pred[1], p_star[1]-p_pred[1]))

        # Predict for plotting
        lb = X_star.min(0)
        ub = X_star.max(0)
        nn = 200
        x = np.linspace(lb[0], ub[0], nn)
        y = np.linspace(lb[1], ub[1], nn)
        X, Y = np.meshgrid(x,y)

        UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
        VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
        PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
        P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')

        if do_noisy_data_case:
            ######################################################################
            ########################### Noisy Data ###############################
            ######################################################################
            noise = 0.01
            u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
            v_train = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])

            # Training
            # model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
            model.train(number_of_training_iterations)  # 200000

            lambda_1_value_noisy = model.sess.run(model.lambda_1)
            lambda_2_value_noisy = model.sess.run(model.lambda_2)

            error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)*100
            error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.01)/0.01 * 100

            print('Error l1: %.5f%%' % (error_lambda_1_noisy))
            print('Error l2: %.5f%%' % (error_lambda_2_noisy))



        ######################################################################
        ############################# Plotting ###############################
        ######################################################################
         # Load Data
        data_vort = scipy.io.loadmat('../Data/cylinder_nektar_t0_vorticity.mat')

        x_vort = data_vort['x']
        y_vort = data_vort['y']
        w_vort = data_vort['w']
        modes = data_vort['modes'].item()
        nel = data_vort['nel'].item()

        xx_vort = np.reshape(x_vort, (modes+1,modes+1,nel), order = 'F')
        yy_vort = np.reshape(y_vort, (modes+1,modes+1,nel), order = 'F')
        ww_vort = np.reshape(w_vort, (modes+1,modes+1,nel), order = 'F')

        box_lb = np.array([1.0, -2.0])
        box_ub = np.array([8.0, 2.0])

        fig, ax = newfig(1.0, 1.2)
        ax.axis('off')

        ####### Row 0: Vorticity ##################
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-2/4 + 0.12, left=0.0, right=1.0, wspace=0)
        ax = plt.subplot(gs0[:, :])

        for i in range(0, nel):
            h = ax.pcolormesh(xx_vort[:,:,i], yy_vort[:,:,i], ww_vort[:,:,i], cmap='seismic',shading='gouraud',  vmin=-3, vmax=3)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)

        ax.plot([box_lb[0],box_lb[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
        ax.plot([box_ub[0],box_ub[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
        ax.plot([box_lb[0],box_ub[0]],[box_lb[1],box_lb[1]],'k',linewidth = 1)
        ax.plot([box_lb[0],box_ub[0]],[box_ub[1],box_ub[1]],'k',linewidth = 1)

        ax.set_aspect('equal', 'box')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('Vorticity', fontsize = 10)


        ####### Row 1: Training data ##################
        ########      u(t,x,y)     ###################
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(top=1-2/4, bottom=0.0, left=0.01, right=0.99, wspace=0)
        ax = plt.subplot(gs1[:, 0],  projection='3d')
        ax.axis('off')

        r1 = [x_star.min(), x_star.max()]
        r2 = [data['t'].min(), data['t'].max()]
        r3 = [y_star.min(), y_star.max()]

        for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
            if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
                ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)

        ax.scatter(x_train, t_train, y_train, s = 0.1)
        ax.contourf(X,UU_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)

        ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
        ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
        ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
        ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$u(t,x,y)$')
        ax.set_xlim3d(r1)
        ax.set_ylim3d(r2)
        ax.set_zlim3d(r3)
        axisEqual3D(ax)

        ########      v(t,x,y)     ###################
        ax = plt.subplot(gs1[:, 1],  projection='3d')
        ax.axis('off')

        r1 = [x_star.min(), x_star.max()]
        r2 = [data['t'].min(), data['t'].max()]
        r3 = [y_star.min(), y_star.max()]

        for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
            if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
                ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)

        ax.scatter(x_train, t_train, y_train, s = 0.1)
        ax.contourf(X,VV_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)

        ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
        ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
        ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
        ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$v(t,x,y)$')
        ax.set_xlim3d(r1)
        ax.set_ylim3d(r2)
        ax.set_zlim3d(r3)
        axisEqual3D(ax)

        # savefig('./figures/NavierStokes_data')


        fig, ax = newfig(1.015, 0.8)
        ax.axis('off')

        ######## Row 2: Pressure #######################
        ########      Predicted p(t,x,y)     ###########
        gs2 = gridspec.GridSpec(1, 2)
        gs2.update(top=1, bottom=1-1/2, left=0.1, right=0.9, wspace=0.5)
        ax = plt.subplot(gs2[:, 0])
        h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow',
                      extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(h, cax=cax)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal', 'box')
        ax.set_title('Predicted pressure', fontsize = 10)

        ########     Exact p(t,x,y)     ###########
        ax = plt.subplot(gs2[:, 1])
        h = ax.imshow(P_exact, interpolation='nearest', cmap='rainbow',
                      extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(h, cax=cax)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal', 'box')
        ax.set_title('Exact pressure', fontsize = 10)


        ######## Row 3: Table #######################
        gs3 = gridspec.GridSpec(1, 2)
        gs3.update(top=1-1/2, bottom=0.0, left=0.0, right=1.0, wspace=0)
        ax = plt.subplot(gs3[:, :])
        ax.axis('off')

        s = r'$\begin{tabular}{|c|c|}';
        s = s + r' \hline'
        s = s + r' Correct PDE & $\begin{array}{c}'
        s = s + r' u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})\\'
        s = s + r' v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})'
        s = s + r' \end{array}$ \\ '
        s = s + r' \hline'
        s = s + r' Identified PDE (clean data) & $\begin{array}{c}'
        s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value, lambda_2_value)
        s = s + r' \\'
        s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value, lambda_2_value)
        s = s + r' \end{array}$ \\ '
        s = s + r' \hline'
        s = s + r' Identified PDE (1\% noise) & $\begin{array}{c}'
        if do_noisy_data_case:
            s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
            s = s + r' \\'
            s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
        s = s + r' \end{array}$ \\ '
        s = s + r' \hline'
        s = s + r' \end{tabular}$'

        ax.text(0.015,0.0,s)

        timer.toc()
        print("Time taken to run: {}".format(timer.elapsed))

        # savefig('./figures/NavierStokes_prediction')
