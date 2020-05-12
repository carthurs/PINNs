import NektarDriver
import ActiveLearningUtilities
import ConfigManager
import SolutionQualityChecker
import NavierStokes
import SimulationParameterManager
import tqdm
import json
import subprocess
import pathlib
import os
import pickle

def get_error_integrals(config_manager, test_vtu_filename_without_extension):
    interprocess_comms_file = pathlib.Path(os.getcwd()) / 'ipc_temp.json'

    try:
        subprocess.run([config_manager.get_paraview_python_interpreter(),
                         'paraviewPythonComputeFieldIntegrals.py',
                         test_vtu_filename_without_extension + '_predicted.vtu',
                         str(interprocess_comms_file)
                         ],
                       timeout=30)
    except subprocess.TimeoutExpired:
        logger = ActiveLearningUtilities.create_logger('SelfTeachingDriver')
        logger.error('Timeout when forming integrals in data file {}'.format(test_vtu_filename_without_extension))

    with open(str(interprocess_comms_file), 'r') as file:
        integrated_errors = json.load(file)
    print('ipc loaded data:', integrated_errors)
    os.remove(str(interprocess_comms_file))

    return integrated_errors


def scatterplot_used_datapoints_and_errors(parameter_manager, test_vtu_filename_template_without_extension,
                                           pickled_model_filename_post, saved_tf_model_filename_post,
                                           true_density, true_viscosity,
                                           config_manager, picklefile_name, logger,
                                           nektar_driver,
                                           parameters_scatter_plot_filename_tag='',
                                           xrange=None,
                                           yrange=None,
                                           colourscale_range=None,
                                           output_subfolder=pathlib.Path('.')):

    integrated_errors_savefile_name = str(output_subfolder /
                                          'integrated_errors_{}.pickle'.format(parameters_scatter_plot_filename_tag))


    if os.path.exists(integrated_errors_savefile_name):
        with open(integrated_errors_savefile_name, 'rb') as infile:
            gathered_error_integrals = pickle.load(infile)
    else:
        gathered_error_integrals = dict()
        for parameters_container in tqdm.tqdm(parameter_manager.all_parameter_points(),
                                              desc='[Gathering error integrals...]',
                                              total=parameter_manager.get_num_parameter_points()):

            test_vtu_filename_without_extension = test_vtu_filename_template_without_extension.format(
                parameters_container.get_t(), parameters_container.get_r())

            test_vtu_filename = test_vtu_filename_without_extension + '.vtu'

            if not os.path.exists(test_vtu_filename):
                nektar_driver.run_simulation(parameters_container)

            NavierStokes.load_and_evaluate_model_against_full_solution(pickled_model_filename_post, saved_tf_model_filename_post,
                                                                       true_density, true_viscosity, test_vtu_filename,
                                                                       parameters_container)

            error_integrals = get_error_integrals(config_manager, test_vtu_filename_without_extension)
            gathered_error_integrals[parameters_container] = error_integrals

        with open(integrated_errors_savefile_name, 'wb') as outfile:
            pickle.dump(gathered_error_integrals, outfile)


    SolutionQualityChecker.scatterplot_parameters_which_have_training_data(picklefile_name,
                                                                           output_filename_tag=parameters_scatter_plot_filename_tag,
                                                                           xrange=xrange, yrange=yrange,
                                                                           output_folder=output_subfolder)

    integrated_errors_to_plot = dict()
    integrated_relative_errors_to_plot = dict()
    error_integral_range = dict()

    field_names = ['u', 'v', 'p']
    for error_integral_field_name in field_names:
        error_integral_range[error_integral_field_name] = [10000, -10000]

    for error_integral_field_name in field_names:
        integrated_errors_to_plot[error_integral_field_name] = dict()
        integrated_relative_errors_to_plot[error_integral_field_name] = dict()
        for parameters_container in parameter_manager.all_parameter_points():
            try:
                error_integral = gathered_error_integrals[parameters_container]['integrals'][error_integral_field_name]
                squared_solution_variable_integral = gathered_error_integrals[parameters_container]['non_error_integrals'][error_integral_field_name]

                error_integral_range[error_integral_field_name][0] = min(error_integral_range[error_integral_field_name][0], error_integral)
                error_integral_range[error_integral_field_name][1] = max(error_integral_range[error_integral_field_name][1], error_integral)

                integrated_errors_to_plot[error_integral_field_name][parameters_container] = error_integral

                if squared_solution_variable_integral < 1E-14:
                    integrated_relative_errors_to_plot[error_integral_field_name][parameters_container] = float('nan')
                else:
                    integrated_relative_errors_to_plot[error_integral_field_name][
                        parameters_container] = error_integral / squared_solution_variable_integral

            except KeyError:
                logger.info('No error data available for parameters {}'.format(parameters_container))

        integrated_errors_for_log = ['{'+str(key)+': '+str(value)+'}' for key, value in integrated_errors_to_plot.items()]
        logger.info("gathered integrated errors were {}".format(''.join(integrated_errors_for_log)))

        # Scatterplot absolute L2 errors
        SolutionQualityChecker.scatterplot_parameters_with_colours(integrated_errors_to_plot[error_integral_field_name],
                                                                   error_integral_field_name,
                                                                   output_filename_tag=parameters_scatter_plot_filename_tag,
                                                                   sim_dir_and_parameter_tuples_picklefile=picklefile_name,
                                                                   colourscale_range=colourscale_range,
                                                                   subfolder_name=output_subfolder)

        # Scatterplot relative L2 errors
        SolutionQualityChecker.scatterplot_parameters_with_colours(integrated_relative_errors_to_plot[error_integral_field_name],
                                                                   error_integral_field_name,
                                                                   output_filename_tag=parameters_scatter_plot_filename_tag+'_relative',
                                                                   sim_dir_and_parameter_tuples_picklefile=picklefile_name,
                                                                   subfolder_name=output_subfolder)

    return error_integral_range


def scatterplot_used_datapoints_and_errors_boundary(parameter_manager, config_manager,
                                                    simulation_parameters_index,
                                                    parameters_scatter_plot_filename_tag='',
                                                    known_training_data_picklefile_name=None,
                                                    colourscale_range=None,
                                                    subfolder_name=''):

    if subfolder_name != '' and not os.path.exists(subfolder_name):
        os.mkdir(subfolder_name)

    integrated_errors_to_plot = dict()
    field_names = ['inflow_velocity_error', 'noslip_pressure', 'noslip_velocity']

    parameters_scatter_plot_filename_tag += '{}'.format(simulation_parameters_index)

    errors_for_this_iteration = NavierStokes.read_boundary_errors(config_manager.get_boundary_errors_filename_template(),
                                                                  simulation_parameters_index)

    for field_name in field_names:
        integrated_errors_to_plot[field_name] = dict()
        for parameters_container in parameter_manager.all_parameter_points():
            integrated_errors_to_plot[field_name][parameters_container] = errors_for_this_iteration[parameters_container][field_name]

    for field_name in field_names:
        SolutionQualityChecker.scatterplot_parameters_with_colours(integrated_errors_to_plot[field_name],
                                                                   field_name,
                                                                   output_filename_tag=parameters_scatter_plot_filename_tag,
                                                                   sim_dir_and_parameter_tuples_picklefile=known_training_data_picklefile_name,
                                                                   colourscale_range=colourscale_range,
                                                                   subfolder_name=pathlib.Path(subfolder_name))


def run_boundary_plotting(simulation_parameters_index, subfolder_name='', colourscale_range=None):
    parameter_descriptor_t = {'range_start': -2.0, 'range_end': 2.0}
    number_of_parameter_points_t = int(
        (parameter_descriptor_t['range_end'] - parameter_descriptor_t['range_start']) * 3) + 1
    parameter_descriptor_t['number_of_points'] = number_of_parameter_points_t

    parameter_descriptor_r = {'range_start': -2.0, 'range_end': 2.0}
    number_of_parameter_points_r = int(
        (parameter_descriptor_r['range_end'] - parameter_descriptor_r['range_start']) * 3) + 1
    parameter_descriptor_r['number_of_points'] = number_of_parameter_points_r

    parameter_manager = SimulationParameterManager.SimulationParameterManager(parameter_descriptor_t,
                                                                              parameter_descriptor_r)

    config_manager = ConfigManager.ConfigManager()
    master_model_data_root_path = config_manager.get_master_model_data_root_path()
    sim_dir_and_parameter_tuples_picklefile_basename = os.path.join(master_model_data_root_path,
                                                                    'sim_dir_and_parameter_tuples_{}start.pickle')
    known_training_data_location_picklefile = sim_dir_and_parameter_tuples_picklefile_basename.format(simulation_parameters_index)

    scatterplot_used_datapoints_and_errors_boundary(parameter_manager, config_manager, simulation_parameters_index,
                                                    known_training_data_picklefile_name=known_training_data_location_picklefile,
                                                    colourscale_range=colourscale_range, subfolder_name=subfolder_name)


def run_plotting(simulation_parameters_index, colourscale_range=None, output_subfolder=pathlib.Path('.')):
    logger = ActiveLearningUtilities.create_logger('SelfTeachingDriver')

    if not os.path.exists(str(output_subfolder)):
        os.mkdir(str(output_subfolder))
        logger.info('Created missing output_subfolder {}'.format(output_subfolder))

    parameter_descriptor_t = {'range_start': -2.0, 'range_end': 2.0}
    number_of_parameter_points_t = int(
        (parameter_descriptor_t['range_end'] - parameter_descriptor_t['range_start']) * 3) + 1
    parameter_descriptor_t['number_of_points'] = number_of_parameter_points_t

    parameter_descriptor_r = {'range_start': -2.0, 'range_end': 2.0}
    number_of_parameter_points_r = int(
        (parameter_descriptor_r['range_end'] - parameter_descriptor_r['range_start']) * 3) + 1
    parameter_descriptor_r['number_of_points'] = number_of_parameter_points_r

    parameter_manager = SimulationParameterManager.SimulationParameterManager(parameter_descriptor_t,
                                                                              parameter_descriptor_r)

    true_viscosity = 0.004
    true_density = 0.00106

    config_manager = ConfigManager.ConfigManager()
    nektar_data_root_path = config_manager.get_nektar_data_root_path()
    reference_data_subfolder = r'basic'
    simulation_subfolder_template = config_manager.get_mesh_data_folder_template()
    vtu_and_xml_file_basename = config_manager.get_vtu_and_xml_file_basename()

    test_vtu_filename_template_without_extension = (nektar_data_root_path + simulation_subfolder_template +
                                                    vtu_and_xml_file_basename + r'_using_points_from_xml')

    master_model_data_root_path = config_manager.get_master_model_data_root_path()
    saved_tf_model_filename_template = os.path.join(master_model_data_root_path, 'saved_model_{}.tf')
    pickled_model_filename_template = os.path.join(master_model_data_root_path, 'saved_model_{}.pickle')

    pickled_model_filename = pickled_model_filename_template.format(simulation_parameters_index)
    saved_tf_model_filename = saved_tf_model_filename_template.format(simulation_parameters_index)

    sim_dir_and_parameter_tuples_picklefile_basename = os.path.join(master_model_data_root_path,
                                                                    'sim_dir_and_parameter_tuples_{}start.pickle')
    picklefile_name = sim_dir_and_parameter_tuples_picklefile_basename.format(simulation_parameters_index)

    # For if FEM test data does not exist yet for any data points - the NektarDriver will be used to generate it
    nektar_driver = NektarDriver.NektarDriver(nektar_data_root_path, reference_data_subfolder,
                                              simulation_subfolder_template,
                                              vtu_and_xml_file_basename,
                                              logger)

    error_integral_range = scatterplot_used_datapoints_and_errors(parameter_manager,
                                                                  test_vtu_filename_template_without_extension,
                                                                  pickled_model_filename, saved_tf_model_filename,
                                                                  true_density, true_viscosity,
                                                                  config_manager, picklefile_name, logger,
                                                                  nektar_driver,
                                                                  parameters_scatter_plot_filename_tag=str(
                                                                      simulation_parameters_index),
                                                                  xrange=(parameter_descriptor_t['range_start'], parameter_descriptor_t['range_end']),
                                                                  yrange=(parameter_descriptor_r['range_start'], parameter_descriptor_r['range_end']),
                                                                  colourscale_range=colourscale_range,
                                                                  output_subfolder=output_subfolder)

    return error_integral_range


if __name__ == '__main__':
    config_manager = ConfigManager.ConfigManager()

    # for step in range(0, 7):
    #     simulation_parameters_index = step * 5 + 1
    #     # run_boundary_plotting(simulation_parameters_index, subfolder_name='boundary_error_plots_ALA_corners', colourscale_range=[0.000001, 0.0001])
    #     run_boundary_plotting(simulation_parameters_index, subfolder_name='boundary_error_plots_ALA_corners_noslip_plots', colourscale_range=[0.00001, 0.0001])

    all_error_integral_ranges = []
    for step in range(0, 12):
        simulation_parameters_index = step * 5 + 1
        error_integral_range = run_plotting(simulation_parameters_index, colourscale_range=[0.1, 10.0],
                                            output_subfolder=config_manager.get_l2_grid_plot_output_subfolder())
        all_error_integral_ranges.append(error_integral_range)

    field_names = ['u', 'v', 'p']
    full_error_range = dict()
    for field_name in field_names:
        full_error_range[field_name] = [10000, -10000]

        for error_range in all_error_integral_ranges:
            full_error_range[field_name][0] = min(full_error_range[field_name][0], error_range[field_name][0])
            full_error_range[field_name][1] = max(full_error_range[field_name][1], error_range[field_name][1])

    print("full error ranges:", full_error_range)