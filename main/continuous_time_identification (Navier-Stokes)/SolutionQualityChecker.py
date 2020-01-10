from NavierStokes import PhysicsInformedNN
import NavierStokes
import VtkDataReader
import tensorflow as tf
import pickle
import numpy as np
import os
import sys

if sys.platform.lower() == "win32":
    os.system('color')

# Group of Different functions for different styles
class style():
    BLACK = lambda x: '\033[30m' + str(x)
    RED = lambda x: '\033[31m' + str(x)
    GREEN = lambda x: '\033[32m' + str(x)
    YELLOW = lambda x: '\033[33m' + str(x)
    BLUE = lambda x: '\033[34m' + str(x)
    MAGENTA = lambda x: '\033[35m' + str(x)
    CYAN = lambda x: '\033[36m' + str(x)
    WHITE = lambda x: '\033[37m' + str(x)
    UNDERLINE = lambda x: '\033[4m' + str(x)
    RESET = lambda x: '\033[0m' + str(x)

def get_losses(pickled_model_filename, saved_tf_model_filename, t_parameter_linspace, plot_id, plot_figures=False):
    with tf.device("/gpu:0"):

        # pickled_model_filename = 'trained_model_nonoise_200000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_10_layers.pickle'
        # saved_tf_model_filename = 'trained_model_nonoise_200000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_10_layers.tf'


        # pickled_model_filename = 'trained_model_nonoise_100000tube10mm_diameter_pt05mesht_gradients_zero_ref_pressure.pickle_10_layers.pickle'
        # saved_tf_model_filename = 'trained_model_nonoise_100000tube10mm_diameter_pt05mesht_gradients_zero_ref_pressure.pickle_10_layers.tf'

        tf.reset_default_graph()
        with open(pickled_model_filename, 'rb') as pickled_model_file:
            model = pickle.load(pickled_model_file)

        model.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                      log_device_placement=False))

        tf.train.Saver().restore(model.sess, saved_tf_model_filename)

        data_file = r'/home/chris/WorkData/nektar++/actual/tube_10mm_diameter_pt2Mesh_correctViscosity/tube10mm_diameter_pt05mesh.vtu'
        # data_file = r'E:\Dev\PINNs\PINNs\main\Data\tube_10mm_diameter_pt2Mesh_correctViscosity\tube10mm_diameter_pt05mesh.vtu'
        data_reader = VtkDataReader.VtkDataReader(data_file, 1.0)
        data = data_reader.get_pinns_format_input_data()
        X_star = data['X_star']  # N x 2
        t_star = data['t']  # T x 1

        N = X_star.shape[0]

        TT = np.tile(t_star, (1, N)).T  # N x T

        x_star = X_star[:, 0:1]
        y_star = X_star[:, 1:2]
        snap = np.array([0])
        t_star = TT[:, snap]

        gathered_losses = dict()
        gathered_boundary_losses = dict()
        for t_parameter in t_parameter_linspace:  #[1, 1.5, 2, 2.5, 3, 4, 3.5, 4.5, 2.2, 1.1, 0.5, -0.5, 5.0, 0.0]:
            t_star = t_star * 0 + t_parameter
            u_pred, v_pred, p_pred, psi_pred = model.predict(x_star, y_star, t_star)

            navier_stokes_loss, boundary_condition_loss = model.get_loss(x_star, y_star, t_star)

            gathered_losses[t_parameter] = navier_stokes_loss
            gathered_boundary_losses[t_parameter] = boundary_condition_loss

            if plot_figures:
                plot_title = "Predicted Velocity U Parameter {} max observed {}".format(t_parameter, np.max(u_pred))
                NavierStokes.plot_solution(X_star, u_pred, plot_id, plot_title, relative_folder_path='./figures_output/')
                plot_id += 1

                plot_title = "Predicted Pressure Parameter {} max observed {}".format(t_parameter, np.max(p_pred))
                NavierStokes.plot_solution(X_star, p_pred, plot_id, plot_title, relative_folder_path='./figures_output/')
                plot_id += 1

        np.savetxt('tempu2.txt', model.get_solution(x_star, y_star, t_star * 0 + 2.5))
        np.savetxt('x.txt', x_star)
        np.savetxt('y.txt', y_star)

    return gathered_losses, gathered_boundary_losses, plot_id


def get_parameter_of_worst_loss(pickled_model_filename, saved_tf_model_filename, t_parameter_linspace, plot_id):
    gathered_losses, gathered_boundary_losses, plot_id = get_losses(pickled_model_filename, saved_tf_model_filename,
                                                                    t_parameter_linspace, plot_id)

    summed_loss = [gathered_losses[loss_1_key] + gathered_boundary_losses[loss_2_key] for
                   (loss_1_key, loss_2_key) in zip(gathered_losses, gathered_boundary_losses)]
    return t_parameter_linspace[np.argmax(summed_loss)], plot_id


def plot_losses(gathered_losses, gathered_boundary_losses, plot_id,
                additional_fig_filename_tag, additional_real_simulation_data_parameters=[]):
    # # this needs to actually be a stored variable not an in-scope variable in the restored class
    # number_of_hidden_layers = 4
    # layers = [3] + [20] * number_of_hidden_layers + [3]
    # parameters_with_real_simulation_data = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] + \
    #                                        additional_real_simulation_data_parameters
    parameters_with_real_simulation_data = additional_real_simulation_data_parameters

    print("Blue indicates that simulation data was used at this point. Yellow indicates interpolation.")
    accuracy_threshold = 0.002
    for key in sorted(gathered_losses):
        value = gathered_losses[key]
        if value > accuracy_threshold:
            value = style.RED(value)
        else:
            value = style.GREEN(value)
        if key in parameters_with_real_simulation_data:
            print(style.BLUE("%.2f" % key), value)
        else:
            print(style.YELLOW("%.2f" % key), value)

    x_data = list(gathered_losses.keys())
    y_data = list(gathered_losses.values())
    second_panel_y_data = list(gathered_boundary_losses.values())
    scatter_x = parameters_with_real_simulation_data
    scatter_y = [gathered_losses[v] for v in scatter_x]
    y_axis_range = (1e-8, 1e4)
    NavierStokes.plot_graph(x_data, y_data, plot_id, 'Loss over Parameters', scatter_x, scatter_y,
                            additional_fig_filename_tag, second_panel_y_data, y_range_1=y_axis_range,
                            y_range_2=y_axis_range, y_range_3=y_axis_range, relative_folder_path='./figures_output/')
    plot_id += 1
    return plot_id


def compute_and_plot_losses(plot_all_figures, pickled_model_filename, saved_tf_model_filename, t_parameter_linspace,
                            plot_id_in, additional_real_simulation_data_parameters=[], plot_filename_tag='1'):
    gathered_losses, gathered_boundary_losses, plot_id = get_losses(pickled_model_filename, saved_tf_model_filename,
                                                                    t_parameter_linspace, plot_id_in, plot_figures=plot_all_figures)

    plot_id = plot_losses(gathered_losses, gathered_boundary_losses, plot_id, plot_filename_tag,
                          additional_real_simulation_data_parameters)
    return plot_id


if __name__ == '__main__':

    plot_all_figures = False
    saved_tf_model_filename = 'retrained4_retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.tf'
    pickled_model_filename = 'retrained4_retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.pickle'
    plot_id_in = 10

    t_parameter_linspace = np.linspace(0.0, 6.0, num=61)
    compute_and_plot_losses(plot_all_figures, pickled_model_filename, saved_tf_model_filename, t_parameter_linspace, plot_id_in)
