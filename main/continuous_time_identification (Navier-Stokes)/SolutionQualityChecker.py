from NavierStokes import PhysicsInformedNN
import NavierStokes
import VtkDataReader
import tensorflow as tf
import pickle
import numpy as np
import os
import sys
import tqdm
import SimulationParameterManager as SPM

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


class LossLandscape(object):
    def __init__(self, pickled_model_filename, saved_tf_model_filename, parameter_manager,
                 data_dir_in, vtu_file_name_template):
        self.pickled_model_filename = pickled_model_filename
        self.saved_tf_model_filename = saved_tf_model_filename
        self.parameter_manager = parameter_manager
        self.data_dir_in = data_dir_in
        self.vtu_file_name_template = vtu_file_name_template

        parameter_space_grid = parameter_manager.get_parameter_space()


        data_shape = parameter_space_grid[0].shape
        self.num_parameter_points = 1
        for parameters_in_this_dimension in data_shape:
            self.num_parameter_points *= parameters_in_this_dimension

        num_parameters = 2

        # Add another column of size 1 for the actual loss value at that point. Rows are points in parameter space, the final column is the loss
        # at that point, and the other columns are the model parameters for that point.
        self.data = np.zeros(shape=(self.num_parameter_points, num_parameters + 1))
        self.next_empty_row_index = 0
        self.landscape_needs_recomputing = True

    class NoAvailableParameters(Exception):
        pass

    def _insert_loss_point(self, parameters_tuple, loss):

        if self.next_empty_row_index >= self.num_parameter_points:
            raise RuntimeError("Tried to insert a point beyond the end of the data array. next_empty_row_index={}, num_points_in_landscape={}.".format(self.next_empty_row_index, self.num_parameter_points))

        self.data[self.next_empty_row_index, :] = parameters_tuple + (loss,)
        self.next_empty_row_index += 1
        self.landscape_needs_recomputing = True

    def get_worst_loss_parameters(self):
        if self.landscape_needs_recomputing:
            self._compute_loss_landscape()
        worst_loss_row_index = np.argmax(self.data[:, -1])
        return self.data[worst_loss_row_index, 0:-1]

    def _compute_loss_landscape(self):
        gathered_losses, gathered_boundary_losses = get_losses(self.pickled_model_filename, self.saved_tf_model_filename,
                                                                        self.parameter_manager, self.data_dir_in,
                                                                        self.vtu_file_name_template)

        summed_loss = [gathered_losses[loss_1_key] + gathered_boundary_losses[loss_2_key] for
                       (loss_1_key, loss_2_key) in zip(gathered_losses, gathered_boundary_losses)]

        # loss_landscape = LossLandscape(len(t_parameter_linspace), 1)
        for index, parameter_container in enumerate(self.parameter_manager.all_parameter_points()):
            self._insert_loss_point((parameter_container.get_t(), parameter_container.get_r()), summed_loss[index])

        return

    def get_parameters_of_worst_loss_excluding_those_near_existing_simulations(self, excluded_parameters, exclusion_radius):
        if self.landscape_needs_recomputing:
            self._compute_loss_landscape()

        row_exclusion_mask = [True] * self.data.shape[0]
        for index in range(self.data.shape[0]):
            param_value_container = SPM.SimulationParameterContainer(self.data[index, 0], self.data[index, 1])
            for excluded_parameter in excluded_parameters:
                if param_value_container.near(excluded_parameter, exclusion_radius):
                    row_exclusion_mask[index] = False
                    break

        no_parameters_available = (max(row_exclusion_mask) == False)
        if no_parameters_available:
            raise LossLandscape.NoAvailableParameters

        excluded_data = self.data[row_exclusion_mask, :]

        worst_loss_row_index = np.argmax(excluded_data[:, -1])
        worst_loss_parameters = excluded_data[worst_loss_row_index, 0:-1]
        return SPM.SimulationParameterContainer(worst_loss_parameters[0], worst_loss_parameters[1])


def get_mesh_embedded_in_regular_grid(mesh_filename, cached_data_dir, parameter_container):
    # data_file = r'/home/chris/WorkData/nektar++/actual/tube_10mm_diameter_pt2Mesh_correctViscosity/tube10mm_diameter_pt05mesh.vtu'
    # data_file = r'E:\Dev\PINNs\PINNs\main\Data\tube_10mm_diameter_pt2Mesh_correctViscosity\tube10mm_diameter_pt05mesh.vtu'
    data_reader = VtkDataReader.VtkDataReader(mesh_filename, parameter_container, cached_data_dir)
    data = data_reader.get_data_by_mode(VtkDataReader.VtkDataReader.MODE_STRUCTURED)
    X_star = data['X_star']  # N x 2
    t_star = data['t']  # T x 1
    r_star = data['r']  # 1 x 1

    N = X_star.shape[0]

    TT = np.tile(t_star, (1, N)).T  # N x T
    RR = np.tile(r_star, (1, N)).T  # N x T

    snap = np.array([0])
    t_star_out = TT[:, snap]
    r_star_out = RR[:, snap]

    return X_star, t_star_out, r_star_out


def plot_on_regular_grid(data_file, data_directory, parameter_container, model, figure_path):
    X_star, t_star, r_star = get_mesh_embedded_in_regular_grid(data_file, data_directory, parameter_container)

    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]

    t_star = t_star * 0 + parameter_container.get_t()
    r_star = r_star * 0 + parameter_container.get_r()
    u_pred, v_pred, p_pred, psi_pred = model.predict(x_star, y_star, t_star, r_star)

    plot_title = "Predicted Velocity U t {} r {} max observed {}".format(parameter_container.get_t(),
                                                                         parameter_container.get_r(),
                                                                         np.max(u_pred))
    NavierStokes.plot_solution(X_star, u_pred, plot_title, relative_or_absolute_folder_path=figure_path)

    plot_title = "Predicted Pressure Parameter t {} r {} max observed {}".format(parameter_container.get_t(),
                                                                                 parameter_container.get_r(),
                                                                                 np.max(p_pred))
    NavierStokes.plot_solution(X_star, p_pred, plot_title, relative_or_absolute_folder_path=figure_path)

    return


def get_losses(pickled_model_filename, saved_tf_model_filename, parameter_manager,
               data_directory, vtu_file_name_template,  plot_figures=False, figure_path='./figures_output/'):
    with tf.device("/gpu:0"):

        tf.reset_default_graph()
        with open(pickled_model_filename, 'rb') as pickled_model_file:
            model = pickle.load(pickled_model_file)

        model.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                      log_device_placement=False))

        tf.train.Saver().restore(model.sess, saved_tf_model_filename)

        gathered_losses = dict()
        gathered_boundary_losses = dict()
        parameters_per_point = parameter_manager.get_parameter_dimensionality()
        for parameter_point_container in tqdm.tqdm(parameter_manager.all_parameter_points(),
                                                   desc='[Computing loss over param space]',
                                                   total=parameter_manager.get_num_parameter_points()):

            data_file = vtu_file_name_template.format(parameter_point_container.get_t(), parameter_point_container.get_r())
            data_reader = VtkDataReader.VtkDataReader(data_file, parameter_point_container, data_directory)
            data = data_reader.get_data_by_mode(VtkDataReader.VtkDataReader.MODE_UNSTRUCTURED)

            X_star = data['X_star']  # N x 2
            x_star = X_star[:, 0:1]
            y_star = X_star[:, 1:2]

            N = X_star.shape[0]
            t_star = np.tile(data['t'], (1, N)).T  # converts the t data from shape 1 x 1 to shape N x 1
            r_star = np.tile(data['r'], (1, N)).T  # converts the r data from shape 1 x 1 to shape N x 1

            boundary_condition_codes = data['bc_codes']

            navier_stokes_loss, boundary_condition_loss = model.get_loss(x_star,
                                                                         y_star,
                                                                         t_star,
                                                                         r_star,
                                                                         boundary_condition_codes)

            gathered_losses[parameter_point_container] = navier_stokes_loss
            gathered_boundary_losses[parameter_point_container] = boundary_condition_loss

            if plot_figures:
                plot_on_regular_grid(data_file, data_directory, parameter_point_container, model, figure_path)
                # TODO could call my non-regular grid plotting tool here to do it on the unstructured grid

    return gathered_losses, gathered_boundary_losses


def plot_losses(gathered_losses, gathered_boundary_losses,
                additional_fig_filename_tag, additional_real_simulation_data_parameters=[],
                figure_path='./figures_output/'):
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
            print(style.BLUE(key), value)
        else:
            print(style.YELLOW(key), value)

    print(style.RESET("Resetting terminal colours..."))

    x_data = range(len(gathered_losses))  # sorted(gathered_losses)
    y_data = [gathered_losses[x] for x in sorted(gathered_losses)]
    second_panel_y_data = list(gathered_boundary_losses.values())
    scatter_x = []
    for index, key in enumerate(sorted(gathered_losses)):
        if key in parameters_with_real_simulation_data:
            scatter_x.append(index)
    scatter_y = [gathered_losses[v] for v in parameters_with_real_simulation_data]
    y_axis_range = (1e-6, 1e8)
    NavierStokes.plot_graph(x_data, y_data, 'Loss over Parameters', scatter_x, scatter_y,
                            additional_fig_filename_tag, second_panel_y_data, y_range_1=y_axis_range,
                            y_range_2=y_axis_range, y_range_3=y_axis_range,
                            relative_or_absolute_output_folder=figure_path)
    return


def compute_and_plot_losses(plot_all_figures, pickled_model_filename, saved_tf_model_filename, parameter_manager,
                            data_dir_in, vtu_file_name_template,
                            additional_real_simulation_data_parameters=(), plot_filename_tag='1'):

    gathered_losses, gathered_boundary_losses = get_losses(pickled_model_filename, saved_tf_model_filename,
                                                                    parameter_manager,
                                                                    data_dir_in, vtu_file_name_template,
                                                                    plot_figures=plot_all_figures,
                                                                    figure_path='{}/figures_output/'.format(data_dir_in))

    plot_losses(gathered_losses,
                  gathered_boundary_losses,
                  plot_filename_tag,
                  additional_real_simulation_data_parameters,
                  figure_path='{}/figures_output/'.format(data_dir_in))
    return


if __name__ == '__main__':

    plot_all_figures = True
    # saved_tf_model_filename = 'retrained4_retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.tf'
    # pickled_model_filename = 'retrained4_retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.pickle'
    model_index_to_load = 6
    data_root = '/home/chris/WorkData/nektar++/actual/bezier/master_data/'
    saved_tf_model_filename = os.path.join(data_root, 'saved_model_{}.tf'.format(model_index_to_load))
    pickled_model_filename = os.path.join(data_root, 'saved_model_{}.pickle'.format(model_index_to_load))

    parameter_manager = SPM.SimulationParameterManager(0.0, 6.0, 61)

    t_parameter_linspace = np.linspace(0.0, 6.0, num=61)
    compute_and_plot_losses(plot_all_figures, pickled_model_filename, saved_tf_model_filename, parameter_manager,
                            data_root)
