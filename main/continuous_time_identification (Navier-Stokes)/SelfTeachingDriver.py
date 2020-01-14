import NektarDriver
import NavierStokes
from NavierStokes import PhysicsInformedNN
import SolutionQualityChecker
import numpy as np
import logging
import pickle

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

    nektar_data_root_path = r'/home/chris/WorkData/nektar++/actual/coarser/'
    reference_data_subfolder = r'tube_10mm_diameter_1pt0Mesh_correctViscosity'

    N_train_in = 10000
    test_mode = False
    if not test_mode:
        num_training_iterations = 20000
        max_optimizer_iterations = 50000
    else:
        num_training_iterations = 20
        max_optimizer_iterations = 50

    use_pressure_node_in_training = True
    vtu_data_file_name = 'tube10mm_diameter_pt05mesh'
    number_of_hidden_layers = 8

    starting_index = 0
    ending_index = 100
    sim_dir_and_parameter_tuples_picklefile_basename = 'sim_dir_and_parameter_tuples_{}start.pickle'
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

        # saved_tf_model_filename = 'retrained{}_retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.tf'
        # pickled_model_filename = 'retrained{}_retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.pickle'
        saved_tf_model_filename = 'saved_model_{}.tf'
        pickled_model_filename = 'saved_model_{}.pickle'

        t_parameter_linspace = np.linspace(0.0, 6.0, num=61)
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
                t_parameter_linspace, plot_id)

            logger.info('Worst loss was at t={}'.format(t_parameter))



            # Ensure that we're not repeating a previously-done simulation, by cutting a hole in the permitted
            # parameter space around the suggested t_parameter, if we already have training data for that parameter.
            exclusion_neighbourhood = np.pi/10  # just something irrational so we don't bump into other values
            t_parameter_linspace_reduced = t_parameter_linspace
            while t_parameter in additional_t_parameters_NS_simulations_run_at:
                t_parameter_linspace_reduced_cached = t_parameter_linspace_reduced

                t_parameter_linspace_reduced = [v for v in t_parameter_linspace_reduced_cached
                                                if abs(v-t_parameter) > exclusion_neighbourhood]

                t_parameter, plot_id = SolutionQualityChecker.get_parameter_of_worst_loss(
                    pickled_model_filename.format(input_data_save_file_tag),
                    saved_tf_model_filename.format(input_data_save_file_tag),
                    t_parameter_linspace_reduced, plot_id)


            start_from_existing_model = True


        info_string = 'Will get data for parameter value t={}'.format(t_parameter)
        logger.info(info_string)

        vtu_file_name = 'tube10mm_diameter_1pt0mesh.vtu'
        nektar_driver = NektarDriver.NektarDriver(nektar_data_root_path, reference_data_subfolder, t_parameter,
                                                  vtu_file_name)
        nektar_driver.run_simulation()

        sim_dir_and_parameter_tuples.append(nektar_driver.get_vtu_file_without_extension_and_parameter())

        picklefile_name = sim_dir_and_parameter_tuples_picklefile_basename.format(input_data_save_file_tag+1)
        with open(picklefile_name, 'wb') as outfile:
            pickle.dump(sim_dir_and_parameter_tuples, outfile)
            logger.info("Saved sim_dir_and_parameter_tuples to file {}".format(picklefile_name))

        NavierStokes.run(pickled_model_filename, saved_tf_model_filename, input_data_save_file_tag, num_training_iterations,
            use_pressure_node_in_training, vtu_data_file_name, number_of_hidden_layers, max_optimizer_iterations,
                         N_train_in, load_existing_model=start_from_existing_model,
                         additional_simulation_data=sim_dir_and_parameter_tuples, parent_logger=logger)

        additional_t_parameters_NS_simulations_run_at = [pair[1] for pair in sim_dir_and_parameter_tuples]

        plot_all_figures = False
        saved_tf_model_filename_post = saved_tf_model_filename.format(input_data_save_file_tag+1)
        pickled_model_filename_post = pickled_model_filename.format(input_data_save_file_tag+1)

        SolutionQualityChecker.compute_and_plot_losses(plot_all_figures, pickled_model_filename_post,
                                                       saved_tf_model_filename_post, t_parameter_linspace,
                                                       plot_id, additional_t_parameters_NS_simulations_run_at,
                                                       plot_filename_tag=str(new_data_iteration))