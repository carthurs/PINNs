import tensorflow as tf
import pickle
import os
import numpy as np
import plotly.express as px
import pandas

import SolutionQualityChecker
import SimulationParameterManager as spm
import ConfigManager
import NektarDriver
import ActiveLearningUtilities

def generate_test_mesh(parameter_container, nektar_data_root_path, reference_data_subfolder, simulation_subfolder_template,
             vtu_and_xml_file_basename, logger):

    nektar_driver = NektarDriver.NektarDriver(nektar_data_root_path, reference_data_subfolder, simulation_subfolder_template,
             vtu_and_xml_file_basename, logger)

    nektar_driver.generate_vtu_mesh_for_parameter(parameter_container)

def get_saved_model_filenames(model_index_to_load):
    data_root = '/home/chris/WorkData/nektar++/actual/bezier/master_data/'
    pickled_model_filename = os.path.join(data_root, 'saved_model_{}.pickle'.format(model_index_to_load))
    saved_tf_model_filename = os.path.join(data_root, 'saved_model_{}.tf'.format(model_index_to_load))

    return pickled_model_filename, saved_tf_model_filename

if __name__ == '__main__':

    # test_parameters = spm.SimulationParameterContainer(t, r)
    get_max_velocity_across_many_parameters = True
    plot_solution = False

    all_test_parameters = []
    # all_test_parameters.append(spm.SimulationParameterContainer(0.68, 0.0))
    all_t_values = np.linspace(-0.2, 1.2, 40)
    for t_value in all_t_values:
        all_test_parameters.append(spm.SimulationParameterContainer(t_value, 0.0))

    config_manager = ConfigManager.ConfigManager()

    nektar_data_root_path = config_manager.get_nektar_data_root_path()
    reference_data_subfolder = r'basic'
    simulation_subfolder_template = config_manager.get_mesh_data_folder_template()
    master_model_data_root_path = config_manager.get_master_model_data_root_path()
    vtu_and_xml_file_basename = config_manager.get_vtu_and_xml_file_basename()
    reference_vtu_filename_template = nektar_data_root_path + simulation_subfolder_template + \
                                      vtu_and_xml_file_basename + '.vtu'

    figure_path = './parameter_point_tests/'

    my_logger = ActiveLearningUtilities.create_logger('TestDataLogger')


    with tf.device("/gpu:0"):

        tf.reset_default_graph()

        pickled_model_filename, saved_tf_model_filename = get_saved_model_filenames(15)

        with open(pickled_model_filename, 'rb') as pickled_model_file:
            model = pickle.load(pickled_model_file)

        model.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                      log_device_placement=False))

        tf.train.Saver().restore(model.sess, saved_tf_model_filename)

        max_velocity_list = []
        for test_parameters in all_test_parameters:
            mesh_filename = reference_vtu_filename_template.format(test_parameters.get_t(), test_parameters.get_r())
            generate_test_mesh(test_parameters, nektar_data_root_path, reference_data_subfolder,
                               simulation_subfolder_template,
                               vtu_and_xml_file_basename, my_logger)

            if plot_solution:
                SolutionQualityChecker.plot_on_regular_grid(mesh_filename, None, test_parameters, model, figure_path)

            if get_max_velocity_across_many_parameters:
                max_velocity = SolutionQualityChecker.get_max_velocity(mesh_filename, None, test_parameters, model)
                max_velocity_list.append(max_velocity)

    data_t_values = [0.0, 0.333, 0.666, 1.0]
    colours = ['no simulation data'] * len(all_t_values) + ['simulation data'] * len(data_t_values)
    frame_t_data = all_t_values.tolist() + data_t_values
    frame_v_data = max_velocity_list + data_t_values
    plot_data = pandas.DataFrame({'inflow parameter': frame_t_data,
                                  'max velocity': frame_v_data,
                                  'point type': colours})
    print(plot_data)
    px.scatter(plot_data, x='inflow parameter', y='max velocity', color='point type').show()
