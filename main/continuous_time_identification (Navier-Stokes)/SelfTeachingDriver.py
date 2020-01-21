import NektarDriver
import NavierStokes
from NavierStokes import PhysicsInformedNN
import SolutionQualityChecker
import numpy as np
import logging
import pickle
import os
import sys

def create_logger():
    logger = logging.getLogger('SelfTeachingDriver')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_log_handler = logging.FileHandler('SelfTeachingDriverLog.txt')
    file_log_handler.setLevel(logging.INFO)
    file_log_handler.setFormatter(formatter)
    logger.addHandler(file_log_handler)

    return logger

if __name__ == '__main__':

    logger = create_logger()
    logger.info("=============== Starting SelfTeachingDriver.py ===============")

    nektar_data_root_path = r'/home/chris/WorkData/nektar++/actual/bezier/'
    # nektar_data_root_path = r'/home/chris/WorkData/nektar++/actual/coarser/'
    reference_data_subfolder = r'basic'
    # reference_data_subfolder = r'tube_10mm_diameter_1pt0Mesh_correctViscosity'
    simulation_subfolder_template = reference_data_subfolder + r'_t{}/'
    master_model_data_root_path = r'/home/chris/WorkData/nektar++/actual/bezier/master_data/'
    # master_model_data_root_path = r'/home/chris/workspace/PINNs/PINNs/main/continuous_time_identification (Navier-Stokes)'
    vtu_and_xml_file_basename = 'tube_bezier_1pt0mesh'
    # vtu_file_name = 'tube10mm_diameter_1pt0mesh'
    reference_vtu_filename_template = nektar_data_root_path + simulation_subfolder_template + \
                                        vtu_and_xml_file_basename + '.vtu'

    use_pressure_node_in_training = True
    number_of_hidden_layers = 8

    starting_index = 0
    ending_index = 100
    sim_dir_and_parameter_tuples_picklefile_basename = os.path.join(master_model_data_root_path,
                                                                    'sim_dir_and_parameter_tuples_{}start.pickle')

    N_train_in = 5000
    test_mode = False
    if not test_mode:
        num_training_iterations = 20000
        max_optimizer_iterations = 50000
        parameter_range_start = 0.0
        parameter_range_end = 6.0
    else:
        num_training_iterations = 20
        max_optimizer_iterations = 50
        parameter_range_start = 0.0
        parameter_range_end = 0.4

    number_of_parameter_points = int((parameter_range_end - parameter_range_start) * 10) + 1

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

    t_parameter_linspace = np.linspace(parameter_range_start, parameter_range_end, num=number_of_parameter_points)
    nektar_driver = NektarDriver.NektarDriver(nektar_data_root_path, reference_data_subfolder,
                                              simulation_subfolder_template,
                                              vtu_and_xml_file_basename,
                                              logger)
    # Ensure that there are meshes for all the t-parameters we're about to query for:
    for t_value in t_parameter_linspace:
        logger.info("Creating mesh for evaluating the predicted solution when parameter={}".format(t_value))
        nektar_driver.generate_vtu_mesh_for_parameter(t_value)

    for new_data_iteration in range(starting_index, ending_index):
        logger.info('Starting iteration {}'.format(new_data_iteration))
        input_data_save_file_tag = new_data_iteration
        logger.info('Nametag is {}'.format(input_data_save_file_tag))

        saved_tf_model_filename = os.path.join(master_model_data_root_path, 'saved_model_{}.tf')
        pickled_model_filename = os.path.join(master_model_data_root_path, 'saved_model_{}.pickle')

        if new_data_iteration == 0:
            # If this is the first iteration, no data is available yet, so we just work with the parameter at the
            # midpoint of the parameter range of interest.
            midpoint = int(len(t_parameter_linspace) / 2)
            t_parameter = t_parameter_linspace[midpoint]
            start_from_existing_model = False
        else:
            logger.info('Will load tensorflow file {}'.format(saved_tf_model_filename.format(input_data_save_file_tag)))
            logger.info('Will load pickle file {}'.format(pickled_model_filename.format(input_data_save_file_tag)))

            loss_landscape = SolutionQualityChecker.LossLandscape(
                pickled_model_filename.format(input_data_save_file_tag),
                saved_tf_model_filename.format(input_data_save_file_tag),
                t_parameter_linspace, master_model_data_root_path,
                reference_vtu_filename_template, num_parameters_per_point=1)

            # Ensure that we're not repeating a previously-done simulation, by cutting a hole in the permitted
            # parameter space around the suggested t_parameter, if we already have training data for that parameter.
            exclusion_radius = np.pi / 10  # just something irrational so we don't bump into other values
            try:
                t_parameter = loss_landscape.get_parameters_of_worst_loss_excluding_those_near_existing_simulations(additional_t_parameters_NS_simulations_run_at,
                                                                                                                    exclusion_radius)[0]
                additional_t_parameters_NS_simulations_run_at.append(t_parameter)
            except SolutionQualityChecker.LossLandscape.NoAvailableParameters:
                logger.info("Could not find another parameter value to simulate at. Parameter space is saturated;\
                                                 no further simulations possible with an exclusion_neighbourhood \
                                                 of size {}."
                            .format(exclusion_radius))
                sys.exit(0)

            start_from_existing_model = True

        info_string = 'Will get data for parameter value t={}'.format(t_parameter)
        logger.info(info_string)

        nektar_driver = NektarDriver.NektarDriver(nektar_data_root_path, reference_data_subfolder,
                                                  simulation_subfolder_template,
                                                  vtu_and_xml_file_basename,
                                                  logger)
        nektar_driver.run_simulation(t_parameter)

        sim_dir_and_parameter_tuples.append((nektar_driver.get_vtu_file_without_extension(t_parameter), t_parameter))

        picklefile_name = sim_dir_and_parameter_tuples_picklefile_basename.format(input_data_save_file_tag+1)
        with open(picklefile_name, 'wb') as outfile:
            pickle.dump(sim_dir_and_parameter_tuples, outfile)
            logger.info("Saved sim_dir_and_parameter_tuples to file {}".format(picklefile_name))

        NavierStokes.run_NS_trainer(pickled_model_filename, saved_tf_model_filename, input_data_save_file_tag,
                                    num_training_iterations, use_pressure_node_in_training, number_of_hidden_layers,
                                    max_optimizer_iterations, N_train_in, load_existing_model=start_from_existing_model,
                                    additional_simulation_data=sim_dir_and_parameter_tuples, parent_logger=logger,
                                    data_caching_directory=master_model_data_root_path)

        additional_t_parameters_NS_simulations_run_at = [pair[1] for pair in sim_dir_and_parameter_tuples]

        plot_all_figures = True
        saved_tf_model_filename_post = saved_tf_model_filename.format(input_data_save_file_tag+1)
        pickled_model_filename_post = pickled_model_filename.format(input_data_save_file_tag+1)

        SolutionQualityChecker.compute_and_plot_losses(plot_all_figures, pickled_model_filename_post,
                                                       saved_tf_model_filename_post, t_parameter_linspace,
                                                       master_model_data_root_path,
                                                       reference_vtu_filename_template,
                                                       additional_real_simulation_data_parameters=additional_t_parameters_NS_simulations_run_at,
                                                       plot_filename_tag=str(new_data_iteration))