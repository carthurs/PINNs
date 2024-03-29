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
import matplotlib.colors
import pathlib
from pytictoc import TicToc
import ActiveLearningUtilities

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
        gathered_losses, gathered_boundary_losses, gathered_total_losses = get_losses(self.pickled_model_filename,
                                                                                      self.saved_tf_model_filename,
                                                                                      self.parameter_manager,
                                                                                      self.data_dir_in,
                                                                                      self.vtu_file_name_template)

        # summed_loss = [gathered_losses[loss_1_key] + gathered_boundary_losses[loss_2_key] for
        #                (loss_1_key, loss_2_key) in zip(gathered_losses, gathered_boundary_losses)]

        # loss_landscape = LossLandscape(len(t_parameter_linspace), 1)
        for index, parameter_container in enumerate(self.parameter_manager.all_parameter_points()):
            self._insert_loss_point( (parameter_container.get_t(),
                                      parameter_container.get_r()),
                                    gathered_losses[parameter_container])

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
            raise ActiveLearningUtilities.NoAvailableParameters

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


def get_max_velocity(data_file, data_directory, parameter_container, model):
    X_star, t_star, r_star = get_mesh_embedded_in_regular_grid(data_file, data_directory, parameter_container)

    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]

    t_star = t_star * 0 + parameter_container.get_t()
    r_star = r_star * 0 + parameter_container.get_r()
    u_pred, v_pred, p_pred, psi_pred = model.predict(x_star, y_star, t_star, r_star)

    return np.max(u_pred)


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
        gathered_total_losses = dict()
        for parameter_point_container in tqdm.tqdm(parameter_manager.all_parameter_points(),
                                                   desc='[Computing loss over param space]',
                                                   total=parameter_manager.get_num_parameter_points()):

            data_file = vtu_file_name_template.format(parameter_point_container.get_t(), parameter_point_container.get_r())
            data_reader = VtkDataReader.VtkDataReader(data_file, parameter_point_container, data_directory)
            data = data_reader.get_data_by_mode(VtkDataReader.VtkDataReader.MODE_UNSTRUCTURED)

            X_star = data['X_star']  # N x 2
            x_star = X_star[:, 0:1]
            y_star = X_star[:, 1:2]

            velocity_fem_data_available = data['U_star'] is not None
            if velocity_fem_data_available:
                u_star = data['U_star'][:, 0]
                v_star = data['U_star'][:, 1]
            else:
                u_star = None
                v_star = None

            N = X_star.shape[0]
            t_star = np.tile(data['t'], (1, N)).T  # converts the t data from shape 1 x 1 to shape N x 1
            r_star = np.tile(data['r'], (1, N)).T  # converts the r data from shape 1 x 1 to shape N x 1

            boundary_condition_codes = data['bc_codes']

            computed_loss_dict = model.get_loss(x_star,
                                                y_star,
                                                t_star,
                                                r_star,
                                                u_star,
                                                v_star,
                                                boundary_condition_codes)

            gathered_losses[parameter_point_container] = computed_loss_dict['navier_stokes_loss']
            gathered_boundary_losses[parameter_point_container] = computed_loss_dict['boundary_condition_loss']
            if velocity_fem_data_available:
                gathered_total_losses[parameter_point_container] = computed_loss_dict['total_loss']

            if plot_figures:
                plot_on_regular_grid(data_file, data_directory, parameter_point_container, model, figure_path)
                # TODO could call my non-regular grid plotting tool here to do it on the unstructured grid

    return gathered_losses, gathered_boundary_losses, gathered_total_losses


def plot_losses(gathered_losses, gathered_boundary_losses, gathered_total_losses,
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
    sorted_keys_for_gathered_total_losses = sorted(gathered_losses)
    for key in sorted_keys_for_gathered_total_losses:
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

    x_data = range(len(gathered_losses))  # sorted_keys_for_gathered_total_losses
    y_data = [gathered_losses[x] for x in sorted_keys_for_gathered_total_losses]
    second_panel_y_data = [gathered_boundary_losses[x] for x in sorted_keys_for_gathered_total_losses]
    second_panel_y_data = None

    scatter_x = []
    scatter_y = []
    for index, key in enumerate(sorted_keys_for_gathered_total_losses):
        if key in parameters_with_real_simulation_data:
            scatter_x.append(index)
            scatter_y.append(gathered_losses[sorted_keys_for_gathered_total_losses[index]])

    y_axis_range = (1e0, 1e8)
    NavierStokes.plot_graph(x_data, y_data, 'Loss over Parameter Space at ALA Iteration {}'.format(additional_fig_filename_tag), scatter_x=scatter_x, scatter_y=scatter_y,
                            savefile_nametag=additional_fig_filename_tag, second_y_data=second_panel_y_data, y_range_1=y_axis_range,
                            y_range_2=y_axis_range, y_range_3=y_axis_range,
                            relative_or_absolute_output_folder=figure_path)
    return


def compute_and_plot_losses(plot_all_figures, pickled_model_filename, saved_tf_model_filename, parameter_manager,
                            data_dir_in, vtu_file_name_template,
                            additional_real_simulation_data_parameters=(), plot_filename_tag='1'):

    gathered_losses, gathered_boundary_losses, gathered_total_losses = get_losses(pickled_model_filename,
                                                                                  saved_tf_model_filename,
                                                                                  parameter_manager,
                                                                                  data_dir_in, vtu_file_name_template,
                                                                                  plot_figures=plot_all_figures,
                                                                                  figure_path='{}/figures_output/'.format(
                                                                                      data_dir_in))

    plot_losses(gathered_losses,
                gathered_boundary_losses,
                gathered_total_losses,
                plot_filename_tag,
                additional_real_simulation_data_parameters,
                figure_path='{}/figures_output/'.format(data_dir_in))

    return


def scatterplot_parameters_with_colours(parameter_container_to_colours_dict, fieldname, output_filename_tag='',
                                        xrange=None, yrange=None, sim_dir_and_parameter_tuples_picklefile=None,
                                        colourscale_range=None, subfolder_name=pathlib.Path('.')):
    scatter_x = []
    scatter_y = []
    scatter_colour = []
    for parameter_container in parameter_container_to_colours_dict:
        scatter_x.append(parameter_container.get_t())
        scatter_y.append(parameter_container.get_r())
        scatter_colour.append(parameter_container_to_colours_dict[parameter_container])

    if min(np.isnan(scatter_colour)) == True:
        logger = logging.getLogger('SelfTeachingDriver')
        logger.warning("Not plotting in scatterplot_parameters_with_colours() because all the colour values were nan.")
        return

    plt.figure(90)

    if colourscale_range is None:
        plt.scatter(scatter_x, scatter_y, c=scatter_colour, vmin=min(scatter_colour), vmax=max(scatter_colour),
                    cmap='cividis', s=250, norm=matplotlib.colors.LogNorm())
    else:
        plt.scatter(scatter_x, scatter_y, c=scatter_colour, vmin=colourscale_range[0], vmax=colourscale_range[1],
                    cmap='cividis', s=250,  norm=matplotlib.colors.LogNorm())
    plt.colorbar()

    title_map = {'inflow_velocity_error': 'Inflow Velocity',
                 'noslip_velocity': 'Wall Velocity',
                 'noslip_pressure': 'Wall Pressure'}
    if fieldname in title_map:
        title_tag = title_map[fieldname]
    else:
        title_tag = fieldname.replace('_', ' ')


    plot_title = 'L2 Errors in {}, ALA Iteration {}'.format(title_tag, output_filename_tag)
    plot_title.replace('_', ' ')
    plt.title(plot_title)

    plt.xlabel('Inflow Parameter')
    plt.ylabel('Domain Shape Parameter')

    if xrange is not None:
        plt.xlim(xrange[0], xrange[1])
    if yrange is not None:
        plt.ylim(yrange[0], yrange[1])

    if sim_dir_and_parameter_tuples_picklefile is not None:
        parameters = get_parameters_which_have_training_data(sim_dir_and_parameter_tuples_picklefile)
        params_with_data_scatter_x = parameters['t']
        params_with_data_scatter_y = parameters['r']
        plt.scatter(params_with_data_scatter_x, params_with_data_scatter_y, c='red', s=20)

    figure_savefile = str(subfolder_name /
                          r'plotted_integrated_errors_{}_{}.png'.format(fieldname, output_filename_tag))

    plt.savefig(figure_savefile)
    plt.close()


def get_parameters_which_have_training_data(sim_dir_and_parameter_tuples_picklefile):
    with open(sim_dir_and_parameter_tuples_picklefile, 'rb') as infile:
        sim_dir_and_parameter_tuples = pickle.load(infile)

    output = dict()
    output['t'] = [x[1].get_t() for x in sim_dir_and_parameter_tuples]
    output['r'] = [y[1].get_r() for y in sim_dir_and_parameter_tuples]

    return output


def scatterplot_parameters_which_have_training_data(sim_dir_and_parameter_tuples_picklefile, output_filename_tag='',
                                                    xrange=None, yrange=None, output_folder=pathlib.Path('.')):
    parameters = get_parameters_which_have_training_data(sim_dir_and_parameter_tuples_picklefile)
    scatter_x = parameters['t']
    scatter_y = parameters['r']

    plt.figure(89)

    plt.scatter(scatter_x, scatter_y, c='red')
    plt.title('Training Parameter Values Used')
    plt.xlabel('Inflow Parameter')
    plt.ylabel('Domain Shape Parameter')
    if xrange is not None:
        plt.xlim(xrange[0], xrange[1])
    if yrange is not None:
        plt.ylim(yrange[0], yrange[1])

    figure_savefile = output_folder / r'plotted_parameters{}.png'.format(output_filename_tag)
    plt.savefig(str(figure_savefile))
    plt.close()


class GradientData(object):
    def __init__(self, x_start, y_start, x_end, y_end, t, r):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.t = t
        self.r = r

    def set_start_and_end_indices(self, start, end):
        self.start_index = start
        self.end_index = end

    def get_relative_gradient(self):
        relative_gradient = (self.p_end - self.p_start)/self.p_start
        print('t={}, r={}, relative gradient={}'.format(self.t, self.r, relative_gradient))
        return relative_gradient

    def get_ffr_pressure(self):
        ffr = self.p_end/self.p_start
        print('FFR (Pressure) = {}, (t={}, r={})'.format(ffr, self.t, self.r))
        return ffr

    def get_pressure_drop(self):
        drop = self.p_start - self.p_end
        print('Pressure Drop = {}, (t={}, r={})'.format(drop, self.t, self.r))
        return self.t, self.r, drop

    def grab_data_from_prediction_array(self, prediction_array):
        p = np.squeeze(prediction_array['p_pred'])
        self.p_start = p[self.start_index]
        self.p_end = p[self.end_index]


if __name__ == '__main__':

    timer = TicToc()
    timer.tic()

    plot_all_figures = True
    # saved_tf_model_filename = 'retrained4_retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.tf'
    # pickled_model_filename = 'retrained4_retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.pickle'
    config_manager = ConfigManager.ConfigManager()
    model_index_to_load = '65'
    data_root = config_manager.get_master_model_data_root_path()
    saved_tf_model_filename = os.path.join(data_root, 'saved_model_{}.tf'.format(model_index_to_load))
    pickled_model_filename = os.path.join(data_root, 'saved_model_{}.pickle'.format(model_index_to_load))

    # parameter_range_start = -2.0
    # parameter_range_end = 2.0
    # number_of_parameter_points = int((parameter_range_end - parameter_range_start) * 3) + 1
    #
    # parameter_manager = SPM.SimulationParameterManager(parameter_range_start,
    #                                                    parameter_range_end,
    #                                                    number_of_parameter_points)

    parameter_descriptor_t = {'range_start': config_manager.get_inflow_parameter_range_start(),
                              'range_end': config_manager.get_inflow_parameter_range_end()}
    number_of_parameter_points_t = int(
        (parameter_descriptor_t['range_end'] - parameter_descriptor_t['range_start']) * 3) + 1
    parameter_descriptor_t['number_of_points'] = number_of_parameter_points_t

    parameter_descriptor_r = {'range_start': config_manager.get_diameter_parameter_range_start(),
                              'range_end': config_manager.get_diameter_parameter_range_end()}
    number_of_parameter_points_r = int(
        (parameter_descriptor_r['range_end'] - parameter_descriptor_r['range_start']) * 3) + 1
    parameter_descriptor_r['number_of_points'] = number_of_parameter_points_r

    parameter_manager = SPM.SimulationParameterManager(parameter_descriptor_t,
                                                                              parameter_descriptor_r)

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
    # t = 2.0
    # r = -2.0

    # test_parameters_container = SPM.SimulationParameterContainer(t, r)
    # test_vtu_filename = r'/home/chris/WorkData/nektar++/actual/bezier/basic_t{}_r{}/tube_bezier_1pt0mesh_using_points_from_xml.vtu'.format(t, r)
    #
    # computed_errors = NavierStokes.load_and_evaluate_model_against_full_solution(pickled_model_filename, saved_tf_model_filename,
    #                                                                              max_optimizer_iterations,
    #                                                                              max_optimizer_iterations,
    #                                                                              true_density_value, true_viscosity_value, test_vtu_filename,
    #                                                                              test_parameters_container)

    # Set single inflow rate for parameter sweep
    t_param = 2.0

    # Set range of diameter points to sweep
    all_gradient_data = []
    for r_param in np.linspace(0, -9, num=81):
        all_gradient_data.append(GradientData(5.0, 5.0, 95.0, 5.0, t_param, r_param))

    N = len(all_gradient_data) * 2
    x_points = np.zeros((N, 1))
    y_points = np.zeros((N, 1))
    t_points = np.zeros((N, 1))
    r_points = np.zeros((N, 1))

    for index, gradient_data in enumerate(all_gradient_data):
        data_start_index = 2 * index
        data_end_index = 2 * index + 1

        gradient_data.set_start_and_end_indices(data_start_index, data_end_index)

        x_points[data_start_index] = gradient_data.x_start
        x_points[data_end_index] = gradient_data.x_end

        y_points[data_start_index] = gradient_data.y_start
        y_points[data_end_index] = gradient_data.y_end

        t_points[data_start_index] = gradient_data.t
        t_points[data_end_index] = gradient_data.t

        r_points[data_start_index] = gradient_data.r
        r_points[data_end_index] = gradient_data.r

    point = {'x': x_points, 'y': y_points, 't': t_points, 'r': r_points}
    prediction = NavierStokes.load_and_evaluate_model_at_point(point, pickled_model_filename, saved_tf_model_filename,
                                                               max_optimizer_iterations)
    # print('prediction2', prediction['p_pred'])
    plotting_array_r = []
    plotting_array_pressure_drop = []
    for gradient_data in all_gradient_data:
        gradient_data.grab_data_from_prediction_array(prediction)
        t, r, drop = gradient_data.get_pressure_drop()
        plotting_array_r.append(r)
        plotting_array_pressure_drop.append(drop)

    NavierStokes.plot_graph(plotting_array_r, plotting_array_pressure_drop, 'narrow_t2pt0_55ALA', logplot=False)

    timer.toc()
