import NektarDriver
import NavierStokes
import ConfigManager
import matplotlib
matplotlib.use('Agg')
from NavierStokes import PhysicsInformedNN
import SolutionQualityChecker
import SimulationParameterManager
import LebesgueErrorPlotter
import numpy as np
import pickle
import sys
import os.path
import ActiveLearningUtilities



class TrainingDataCountSpecifier(object):
    PROPORTION = 0
    ABSOLUTE = 1

    def __init__(self, mode, parameter):
        self.mode = mode
        self.parameter = parameter
        self.total_points = None

    def set_num_total_available_datapoints(self, total_points):
        self.total_points = total_points

    def get_number_to_use_in_training(self):
        if self.mode == TrainingDataCountSpecifier.PROPORTION:
            if self.total_points is None:
                raise RuntimeError("Total points not set on this class yet. Please set them before use.")
            if self.parameter <= 0 or self.parameter >= 1:
                raise RuntimeError("In proportional mode, the parameter must be between 0 and 1, exclusive.")
            total_training_points = int(self.parameter * self.total_points)
        elif self.mode == TrainingDataCountSpecifier.ABSOLUTE:
            total_training_points = self.parameter
        else:
            raise RuntimeError("Unknown mode passed during construction of this class.")

        return total_training_points


if __name__ == '__main__':

    logger = ActiveLearningUtilities.create_logger('SelfTeachingDriver')
    logger.info("=============== Starting SelfTeachingDriver.py ===============")
    config_manager = ConfigManager.ConfigManager()

    nektar_data_root_path = config_manager.get_nektar_data_root_path()
    reference_data_subfolder = r'basic'
    simulation_subfolder_template = config_manager.get_mesh_data_folder_template()
    master_model_data_root_path = config_manager.get_master_model_data_root_path()
    vtu_and_xml_file_basename = config_manager.get_vtu_and_xml_file_basename()
    reference_vtu_filename_template = nektar_data_root_path + simulation_subfolder_template + \
                                        vtu_and_xml_file_basename + '.vtu'

    true_viscosity = 0.004
    true_density = 0.00106

    use_pressure_reference_in_training = True
    number_of_hidden_layers = 4

    starting_index = 63
    ending_index = 200
    sim_dir_and_parameter_tuples_picklefile_basename = os.path.join(master_model_data_root_path,
                                                                    'sim_dir_and_parameter_tuples_{}start.pickle')

    training_count_specifier = TrainingDataCountSpecifier(TrainingDataCountSpecifier.PROPORTION, 0.99)
    test_mode = False
    if not test_mode:
        num_training_iterations = 20000
        max_optimizer_iterations = 50000
        parameter_range_start = -2.0
        parameter_range_end = 2.0
    else:
        num_training_iterations = 20
        max_optimizer_iterations = 50
        parameter_range_start = 0.0
        parameter_range_end = 1.0

    number_of_parameter_points = int((parameter_range_end - parameter_range_start) * 3) + 1

    try:
        sim_dir_and_parameter_tuples_picklefile = sim_dir_and_parameter_tuples_picklefile_basename.format(starting_index)
        with open(sim_dir_and_parameter_tuples_picklefile, 'rb') as infile:
            sim_dir_and_parameter_tuples = pickle.load(infile)
        additional_t_parameters_NS_simulations_run_at = [pair[1] for pair in sim_dir_and_parameter_tuples]
        logger.info("Loaded data on previously run simulation iterations from data file {}".format(
                                                                            sim_dir_and_parameter_tuples_picklefile))
    except FileNotFoundError:
        sim_dir_and_parameter_tuples = []
        additional_t_parameters_NS_simulations_run_at = []
        logger.warning("Previous simualtion iterations not found. Starting from scratch at iteration {}".format(
                                                                                                        starting_index))

    parameter_manager = SimulationParameterManager.SimulationParameterManager(parameter_range_start,
                                                                              parameter_range_end,
                                                                              number_of_parameter_points)


    nektar_driver = NektarDriver.NektarDriver(nektar_data_root_path, reference_data_subfolder,
                                              simulation_subfolder_template,
                                              vtu_and_xml_file_basename,
                                              logger)
    # Ensure that there are meshes for all the t-parameters we're about to query for:
    for param_container in parameter_manager.all_parameter_points():
        logger.info("Creating mesh for evaluating the predicted solution when parameters are {}".format(param_container))
        nektar_driver.generate_vtu_mesh_for_parameter(param_container)

    for simulation_parameters_index in range(starting_index, ending_index):
        logger.info('Starting iteration {}'.format(simulation_parameters_index))
        logger.info('Nametag is {}'.format(simulation_parameters_index))

        saved_tf_model_filename = os.path.join(master_model_data_root_path, 'saved_model_{}.tf')
        pickled_model_filename = os.path.join(master_model_data_root_path, 'saved_model_{}.pickle')

        if simulation_parameters_index == 0:
            # If this is the first iteration, no data is available yet, so we just work with the parameter at the
            # midpoint of the parameter range of interest.
            parameters_container = parameter_manager.get_initial_parameters()
            start_from_existing_model = False
        else:
            logger.info('Will load tensorflow file {}'.format(saved_tf_model_filename.format(simulation_parameters_index)))
            logger.info('Will load pickle file {}'.format(pickled_model_filename.format(simulation_parameters_index)))

            loss_landscape = SolutionQualityChecker.LossLandscape(
                pickled_model_filename.format(simulation_parameters_index),
                saved_tf_model_filename.format(simulation_parameters_index),
                parameter_manager, master_model_data_root_path,
                reference_vtu_filename_template)

            # Ensure that we're not repeating a previously-done simulation, by cutting a hole in the permitted
            # parameter space around the suggested t_parameter, if we already have training data for that parameter.
            exclusion_radius = np.pi / 30  # just something irrational so we don't bump into other values
            try:
                parameters_container = loss_landscape.get_parameters_of_worst_loss_excluding_those_near_existing_simulations(additional_t_parameters_NS_simulations_run_at,
                                                                                                                    exclusion_radius)
                # additional_t_parameters_NS_simulations_run_at.append(t_parameter)
            except SolutionQualityChecker.LossLandscape.NoAvailableParameters:
                logger.info("Could not find another parameter value to simulate at. Parameter space is saturated;\
                                                 no further simulations possible with an exclusion_neighbourhood \
                                                 of size {}."
                            .format(exclusion_radius))
                sys.exit(0)

            start_from_existing_model = True

        parameters_container.log()

        nektar_driver = NektarDriver.NektarDriver(nektar_data_root_path, reference_data_subfolder,
                                                  simulation_subfolder_template,
                                                  vtu_and_xml_file_basename,
                                                  logger)
        nektar_driver.run_simulation(parameters_container)

        sim_dir_and_parameter_tuples.append((nektar_driver.get_vtu_file_without_extension(parameters_container), parameters_container))

        picklefile_name = sim_dir_and_parameter_tuples_picklefile_basename.format(simulation_parameters_index + 1)
        with open(picklefile_name, 'wb') as outfile:
            pickle.dump(sim_dir_and_parameter_tuples, outfile)
            logger.info("Saved sim_dir_and_parameter_tuples to file {}".format(picklefile_name))

        NavierStokes.run_NS_trainer(pickled_model_filename, saved_tf_model_filename, simulation_parameters_index,
                                    num_training_iterations, use_pressure_reference_in_training, number_of_hidden_layers,
                                    max_optimizer_iterations, training_count_specifier, true_viscosity,
                                    true_density, load_existing_model=start_from_existing_model,
                                    additional_simulation_data=sim_dir_and_parameter_tuples, parent_logger=logger,
                                    data_caching_directory=master_model_data_root_path)

        SolutionQualityChecker.scatterplot_parameters_which_have_training_data(picklefile_name)

        additional_t_parameters_NS_simulations_run_at = [pair[1] for pair in sim_dir_and_parameter_tuples]

        plot_all_figures = True
        saved_tf_model_filename_post = saved_tf_model_filename.format(simulation_parameters_index + 1)
        pickled_model_filename_post = pickled_model_filename.format(simulation_parameters_index + 1)

        SolutionQualityChecker.compute_and_plot_losses(plot_all_figures, pickled_model_filename_post,
                                                       saved_tf_model_filename_post, parameter_manager,
                                                       master_model_data_root_path,
                                                       reference_vtu_filename_template,
                                                       additional_real_simulation_data_parameters=additional_t_parameters_NS_simulations_run_at,
                                                       plot_filename_tag=str(simulation_parameters_index))

        test_vtu_filename_template_without_extension = (nektar_data_root_path + simulation_subfolder_template +
                                                        vtu_and_xml_file_basename + r'_using_points_from_xml')

        scatterplot_tag = simulation_parameters_index + 1
        LebesgueErrorPlotter.scatterplot_used_datapoints_and_errors(parameter_manager,
                                                                    test_vtu_filename_template_without_extension,
                                                                    pickled_model_filename_post,
                                                                    saved_tf_model_filename_post,
                                                                    true_density, true_viscosity,
                                                                    config_manager, picklefile_name, logger,
                                                                    nektar_driver,
                                                                    parameters_scatter_plot_filename_tag=scatterplot_tag,
                                                                    xrange=(parameter_range_start, parameter_range_end),
                                                                    yrange=(parameter_range_start, parameter_range_end))