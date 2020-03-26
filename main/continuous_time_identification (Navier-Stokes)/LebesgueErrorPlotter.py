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

    subprocess.call([config_manager.get_paraview_python_interpreter(),
                     'paraviewPythonComputeFieldIntegrals.py',
                     test_vtu_filename_without_extension + '_predicted.vtu',
                     str(interprocess_comms_file)
                     ])

    with open(str(interprocess_comms_file), 'r') as file:
        integrated_errors = json.load(file)
    print('ipc loaded data:', integrated_errors)
    os.remove(str(interprocess_comms_file))

    return integrated_errors


def scatterplot_used_datapoints_and_errors(parameter_manager, test_vtu_filename_template_without_extension,
                                           pickled_model_filename_post, saved_tf_model_filename_post,
                                           max_optimizer_iterations,
                                           true_density, true_viscosity,
                                           config_manager, picklefile_name, logger,
                                           nektar_driver,
                                           parameters_scatter_plot_filename_tag='',
                                           xrange=None,
                                           yrange=None):
    gathered_error_integrals = dict()

    for parameters_container in tqdm.tqdm(parameter_manager.all_parameter_points(),
                                          desc='[Gathering error integrals...]',
                                          total=parameter_manager.get_num_parameter_points()):

        test_vtu_filename_without_extension = test_vtu_filename_template_without_extension.format(
            parameters_container.get_t(), parameters_container.get_r())

        test_vtu_filename = test_vtu_filename_without_extension + '.vtu'

        if not os.path.exists(test_vtu_filename):
            nektar_driver.run_simulation(parameters_container)

        NavierStokes.load_and_evaluate_model(pickled_model_filename_post, saved_tf_model_filename_post,
                                             max_optimizer_iterations,
                                             true_density, true_viscosity, test_vtu_filename,
                                             parameters_container)

        error_integrals = get_error_integrals(config_manager, test_vtu_filename_without_extension)
        gathered_error_integrals[parameters_container] = error_integrals


    SolutionQualityChecker.scatterplot_parameters_which_have_training_data(picklefile_name,
                                                                           output_filename_tag=parameters_scatter_plot_filename_tag,
                                                                           xrange=xrange, yrange=yrange)

    integrated_errors_to_plot = dict()
    error_integral_range = [10000, -10000]

    for error_integral_field_name in ['u', 'v', 'p']:
        integrated_errors_to_plot[error_integral_field_name] = dict()
        for parameters_container in parameter_manager.all_parameter_points():
            try:
                error_integral = gathered_error_integrals[parameters_container]['integrals'][error_integral_field_name]

                error_integral_range[0] = min(error_integral_range[0], error_integral)
                error_integral_range[1] = max(error_integral_range[1], error_integral)

                integrated_errors_to_plot[error_integral_field_name][parameters_container] = error_integral
            except KeyError:
                logger.info('No error data available for parameters {}'.format(parameters_container))

        integrated_errors_for_log = ['{'+str(key)+': '+str(value)+'}' for key, value in integrated_errors_to_plot.items()]
        logger.info("gathered integrated errors were {}".format(''.join(integrated_errors_for_log)))

        SolutionQualityChecker.scatterplot_parameters_with_colours(integrated_errors_to_plot[error_integral_field_name],
                                                                   error_integral_field_name,
                                                                   output_filename_tag=parameters_scatter_plot_filename_tag)

    with open('integrated_errors_{}.pickle'.format(parameters_scatter_plot_filename_tag), 'wb') as outfile:
        pickle.dump(integrated_errors_to_plot, outfile)


if __name__ == '__main__':
    logger = ActiveLearningUtilities.create_logger('SelfTeachingDriver')
    parameter_range_start = -2.0
    parameter_range_end = 2.0
    number_of_parameter_points = int((parameter_range_end - parameter_range_start) * 3) + 1

    simulation_parameters_index = 1
    true_viscosity = 0.004
    true_density = 0.00106

    parameter_manager = SimulationParameterManager.SimulationParameterManager(parameter_range_start,
                                                                              parameter_range_end,
                                                                              number_of_parameter_points)
    config_manager = ConfigManager.ConfigManager()
    nektar_data_root_path = config_manager.get_nektar_data_root_path()
    reference_data_subfolder = r'basic'
    simulation_subfolder_template = reference_data_subfolder + r'_t{}_r{}/'
    vtu_and_xml_file_basename = 'tube_bezier_1pt0mesh'

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

    max_optimizer_iterations = 50000
    scatterplot_used_datapoints_and_errors(parameter_manager, test_vtu_filename_template_without_extension,
                                           pickled_model_filename, saved_tf_model_filename,
                                           max_optimizer_iterations,
                                           true_density, true_viscosity,
                                           config_manager, picklefile_name, logger,
                                           nektar_driver,
                                           parameters_scatter_plot_filename_tag=str(simulation_parameters_index),
                                           xrange=(parameter_range_start, parameter_range_end),
                                           yrange=(parameter_range_start, parameter_range_end))