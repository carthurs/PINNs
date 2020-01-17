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
    master_model_data_root_path = r'/home/chris/WorkData/nektar++/actual/bezier/master_data/'
    # master_model_data_root_path = r'/home/chris/workspace/PINNs/PINNs/main/continuous_time_identification (Navier-Stokes)'
    vtu_and_xml_file_basename = 'tube_bezier_1pt0mesh'
    # vtu_file_name = 'tube10mm_diameter_1pt0mesh'
    use_pressure_node_in_training = True
    number_of_hidden_layers = 8

    parameter_range_start = 0.0
    parameter_range_end = 6.0
    number_of_parameter_points = int((parameter_range_end - parameter_range_start)*10) + 1

    starting_index = 6
    ending_index = 100
    sim_dir_and_parameter_tuples_picklefile_basename = os.path.join(master_model_data_root_path,
                                                                    'sim_dir_and_parameter_tuples_{}start.pickle')

    N_train_in = 5000
    test_mode = False
    if not test_mode:
        num_training_iterations = 20000
        max_optimizer_iterations = 50000
    else:
        num_training_iterations = 20
        max_optimizer_iterations = 50

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

    for new_data_iteration in range(starting_index, ending_index):
        logger.info('Starting iteration {}'.format(new_data_iteration))
        input_data_save_file_tag = new_data_iteration
        logger.info('Nametag is {}'.format(input_data_save_file_tag))

        saved_tf_model_filename = os.path.join(master_model_data_root_path, 'saved_model_{}.tf')
        pickled_model_filename = os.path.join(master_model_data_root_path, 'saved_model_{}.pickle')

        t_parameter_linspace = np.linspace(parameter_range_start, parameter_range_end, num=number_of_parameter_points)
        plot_id = 10
        if new_data_iteration == 0:
            # If this is the first iteration, no data is available yet, so we just work with the parameter at the
            # midpoint of the parameter range of interest.
            midpoint = int(len(t_parameter_linspace) / 2)
            t_parameter = t_parameter_linspace[midpoint]
            start_from_existing_model = False
        else:
            logger.info('Will load tensorflow file {}'.format(saved_tf_model_filename.format(input_data_save_file_tag)))
            logger.info('Will load pickle file {}'.format(pickled_model_filename.format(input_data_save_file_tag)))

            t_parameter, plot_id = SolutionQualityChecker.get_parameter_of_worst_loss(
                pickled_model_filename.format(input_data_save_file_tag),
                saved_tf_model_filename.format(input_data_save_file_tag),
                t_parameter_linspace, plot_id, master_model_data_root_path)

            logger.info('Worst loss was at t={}'.format(t_parameter))

            # Ensure that we're not repeating a previously-done simulation, by cutting a hole in the permitted
            # parameter space around the suggested t_parameter, if we already have training data for that parameter.
            exclusion_neighbourhood = np.pi/10  # just something irrational so we don't bump into other values
            t_parameter_linspace_reduced = t_parameter_linspace
            while_loop_counter = 0
            while t_parameter in additional_t_parameters_NS_simulations_run_at:
                t_parameter_linspace_reduced_cached = t_parameter_linspace_reduced

                t_parameter_linspace_reduced = [v for v in t_parameter_linspace_reduced_cached
                                                if abs(v-t_parameter) > exclusion_neighbourhood]

                if len(t_parameter_linspace_reduced) == 0:
                    logger.info("Could not find another parameter value to simulate at. Parameter space is saturated;\
                                 no further simulations possible with an exclusion_neighbourhood of size {}."
                                .format(while_loop_counter, exclusion_neighbourhood))
                    sys.exit(0)

                t_parameter, plot_id = SolutionQualityChecker.get_parameter_of_worst_loss(
                    pickled_model_filename.format(input_data_save_file_tag),
                    saved_tf_model_filename.format(input_data_save_file_tag),
                    t_parameter_linspace_reduced, plot_id, master_model_data_root_path)
                while_loop_counter += 1

            start_from_existing_model = True

        info_string = 'Will get data for parameter value t={}'.format(t_parameter)
        logger.info(info_string)

        nektar_driver = NektarDriver.NektarDriver(nektar_data_root_path, reference_data_subfolder, t_parameter,
                                                  vtu_and_xml_file_basename)
        nektar_driver.run_simulation()

        sim_dir_and_parameter_tuples.append(nektar_driver.get_vtu_file_without_extension_and_parameter())

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
                                                       plot_id, master_model_data_root_path,
                                                       additional_t_parameters_NS_simulations_run_at,
                                                       plot_filename_tag=str(new_data_iteration))