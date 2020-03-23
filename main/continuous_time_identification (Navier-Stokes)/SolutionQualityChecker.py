from NavierStokes import PhysicsInformedNN
import NavierStokes
import VtkDataReader
import logging
import tensorflow as tf
import pickle
import numpy as np
import os
import sys
import tqdm
import SimulationParameterManager as SPM
import multiprocessing
import ConfigManager
import matplotlib.pyplot as plt

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

    info_string = "==== Loaded mesh file {} with parameters r={}, t={}".format(mesh_filename, parameter_container.get_r(), parameter_container.get_t())
    logger = logging.getLogger('SelfTeachingDriver')
    logger.info(info_string)

    logger.info("Data dir was {}".format(cached_data_dir))

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


def _do_plot(parameter_container, X_star, u_pred, p_pred, figure_path):
    stringify_and_shorten = lambda s: str(s)[0:5]
    short_t_string = stringify_and_shorten(parameter_container.get_t())
    short_r_string = stringify_and_shorten(parameter_container.get_r())

    short_max_velocity_string = stringify_and_shorten(np.max(u_pred))
    short_min_velocity_string = stringify_and_shorten(np.min(u_pred))

    plot_title = "Predicted Velocity U t {} r {} max {} min {}".format(short_t_string,
                                                                       short_r_string,
                                                                       short_max_velocity_string,
                                                                       short_min_velocity_string)

    NavierStokes.plot_solution(X_star, u_pred, plot_title, relative_or_absolute_folder_path=figure_path)

    logger = logging.getLogger('SelfTeachingDriver')
    logger.info("plotting with title {}".format(plot_title))

    short_max_pressure_string = stringify_and_shorten(np.max(p_pred))
    short_min_pressure_string = stringify_and_shorten(np.min(p_pred))

    plot_title = "Predicted Pressure Parameter t {} r {} max {} min {}".format(short_t_string,
                                                                               short_r_string,
                                                                               short_max_pressure_string,
                                                                               short_min_pressure_string)

    NavierStokes.plot_solution(X_star, p_pred, plot_title, relative_or_absolute_folder_path=figure_path)


def plot_on_regular_grid(data_file, data_directory, parameter_container, model, figure_path):
    X_star, t_star, r_star = get_mesh_embedded_in_regular_grid(data_file, data_directory, parameter_container)

    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]

    t_star = t_star * 0 + parameter_container.get_t()
    r_star = r_star * 0 + parameter_container.get_r()
    u_pred, v_pred, p_pred, psi_pred = model.predict(x_star, y_star, t_star, r_star)

    p = multiprocessing.Process(target=_do_plot,
                                args=(parameter_container, X_star, u_pred, p_pred, figure_path))
    p.start()
    # p.join()
    # print("exit code was:", p.exitcode)

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
    sorted_keys_for_gathered_losses = sorted(gathered_losses)
    for key in sorted_keys_for_gathered_losses:
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

    x_data = range(len(gathered_losses))  # sorted_keys_for_gathered_losses
    y_data = [gathered_losses[x] for x in sorted_keys_for_gathered_losses]
    second_panel_y_data = list(gathered_boundary_losses.values())
    scatter_x = []
    scatter_y = []
    for index, key in enumerate(sorted_keys_for_gathered_losses):
        if key in parameters_with_real_simulation_data:
            scatter_x.append(index)
            scatter_y.append(gathered_losses[sorted_keys_for_gathered_losses[index]])

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


def scatterplot_parameters_which_have_training_data(sim_dir_and_parameter_tuples_picklefile, output_filename_tag=''):
    with open(sim_dir_and_parameter_tuples_picklefile, 'rb') as infile:
        sim_dir_and_parameter_tuples = pickle.load(infile)

    scatter_x = [x[1].get_t() for x in sim_dir_and_parameter_tuples]
    scatter_y = [y[1].get_r() for y in sim_dir_and_parameter_tuples]

    plt.figure(89)

    plt.scatter(scatter_x, scatter_y, c='red')
    plt.title('Training Parameter Values Used')
    plt.xlabel('Inflow Parameter')
    plt.ylabel('Domain Shape Parameter')

    figure_savefile = r'plotted_parameters{}.png'.format(output_filename_tag)
    plt.savefig(figure_savefile)
    plt.close()


if __name__ == '__main__':

    plot_all_figures = True
    # saved_tf_model_filename = 'retrained4_retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.tf'
    # pickled_model_filename = 'retrained4_retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.pickle'
    model_index_to_load = 19
    data_root = '/home/chris/WorkData/nektar++/actual/bezier/master_data/'
    saved_tf_model_filename = os.path.join(data_root, 'saved_model_{}.tf'.format(model_index_to_load))
    pickled_model_filename = os.path.join(data_root, 'saved_model_{}.pickle'.format(model_index_to_load))

    parameter_range_start = -2.0
    parameter_range_end = 2.0
    number_of_parameter_points = int((parameter_range_end - parameter_range_start) * 3) + 1

    parameter_manager = SPM.SimulationParameterManager(parameter_range_start,
                                                       parameter_range_end,
                                                       number_of_parameter_points)

    # t_parameter_linspace = np.linspace(0.0, 6.0, num=61)

    # compute_and_plot_losses(plot_all_figures, pickled_model_filename, saved_tf_model_filename, parameter_manager,
    #                         data_root)

    config_manager = ConfigManager.ConfigManager()
    master_model_data_root_path = config_manager.get_master_model_data_root_path()
    nektar_data_root_path = config_manager.get_nektar_data_root_path()
    sim_dir_and_parameter_tuples_picklefile_basename = os.path.join(master_model_data_root_path,
                                                                    'sim_dir_and_parameter_tuples_{}start.pickle')

    # scatterplot_parameters_which_have_training_data(sim_dir_and_parameter_tuples_picklefile_basename.format(model_index_to_load))

    plot_lots_in = False
    true_viscosity_value = 0.004  # 0.01
    true_density_value = 0.00106  # 1.0
    max_optimizer_iterations = 50000
    t = 1.6666666666666665
    r = 1.333333333333333

    test_parameters_container = SPM.SimulationParameterContainer(t, r)
    test_vtu_filename = r'/home/chris/WorkData/nektar++/actual/bezier/basic_t{}_r{}/tube_bezier_1pt0mesh_using_points_from_xml.vtu'.format(t, r)

    computed_errors = NavierStokes.load_and_evaluate_model(pickled_model_filename, saved_tf_model_filename,
                                                           max_optimizer_iterations,
                                                           true_density_value, true_viscosity_value, test_vtu_filename,
                                                           test_parameters_container)
