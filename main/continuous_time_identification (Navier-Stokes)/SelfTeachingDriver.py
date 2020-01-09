import NektarDriver
import NavierStokes
from NavierStokes import PhysicsInformedNN
import SolutionQualityChecker
import numpy as np

if __name__ == '__main__':
    nektar_data_root_path = r'/home/chris/WorkData/nektar++/actual/'
    reference_data_subfolder = r'tube_10mm_diameter_pt2Mesh_correctViscosity'

    saved_tf_model_filename = 'retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.tf'
    pickled_model_filename = 'retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.pickle'

    plot_id = 10
    t_parameter_linspace = np.linspace(0.0, 6.0, num=61)
    t_parameter, plot_id = SolutionQualityChecker.get_parameter_of_worst_loss(pickled_model_filename,
                                                                              saved_tf_model_filename,
                                                                              t_parameter_linspace, plot_id)

    print('Will get data for parameter value t={}'.format(t_parameter))

    vtu_file_name = 'tube10mm_diameter_pt05mesh.vtu'
    nektar_driver = NektarDriver.NektarDriver(nektar_data_root_path, reference_data_subfolder, t_parameter, vtu_file_name)
    nektar_driver.run_simulation()

    sim_dir_and_parameter_tuple = nektar_driver.get_vtu_file_without_extension_and_parameter()
    NavierStokes.run(additional_simulation_data=[sim_dir_and_parameter_tuple])

    plot_all_figures = False
    saved_tf_model_filename_post = 'retrained4_retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.tf'
    pickled_model_filename_post = 'retrained4_retrained3_retrained2_retrained_trained_model_nonoise_100000tube10mm_diameter_pt05meshworking_500TrainingDatapoints_zero_ref_pressure.pickle_6_layers.pickle'

    SolutionQualityChecker.compute_and_plot_losses(plot_all_figures, pickled_model_filename_post,
                                                   saved_tf_model_filename_post, t_parameter_linspace,
                                                   plot_id, [t_parameter])