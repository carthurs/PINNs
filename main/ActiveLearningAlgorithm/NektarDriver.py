import os
import distutils.dir_util
import subprocess
import fileinput
import VtkDataReader
import ConfigManager
import SimulationParameterManager
import ActiveLearningUtilities
from pytictoc import TicToc
import Meshing


class NektarDriver(object):
    def __init__(self, nektar_data_root_path, reference_data_subfolder, simulation_subfolder_template,
                 vtu_and_xml_file_basename, logger):
        self.nektar_data_root_path = nektar_data_root_path
        self.reference_data_subfolder = reference_data_subfolder
        self.simulation_subfolder_template = simulation_subfolder_template
        self.vtu_and_xml_file_basename = vtu_and_xml_file_basename
        self.vtu_file_name = self.vtu_and_xml_file_basename + '.vtu'
        self.initial_working_path = os.getcwd()
        self.mesh_xml_file_name = self.vtu_and_xml_file_basename + '.xml'
        self.fld_file_name = self.vtu_and_xml_file_basename + '.fld'
        self.logger = logger
        self.config_manager = ConfigManager.ConfigManager()

    def generate_vtu_mesh_for_parameter(self, param_container):
        self._prepare_simulation_files(param_container, generate_vtu_without_solution_data=True)

    def _prepare_simulation_files(self, parameters_container, generate_vtu_without_solution_data=False):
        generator = Meshing.MeshGenerator(self.mesh_xml_file_name, parameters_container, logger=self.logger)

        if generate_vtu_without_solution_data:
            generator.generate_in_correct_folder(vtu_additional_file_name=self.vtu_file_name)
        else:
            generator.generate_in_correct_folder()

    def run_simulation(self, parameters_container):
        self._prepare_simulation_files(parameters_container)
        os.chdir(self.nektar_data_root_path)
        simulation_subfolder = self.simulation_subfolder_template.format(parameters_container.get_t(),
                                                                         parameters_container.get_r())
        os.chdir(simulation_subfolder)

        # Set the peak inflow velocity
        self._set_simulation_inflow_parameter(parameters_container.get_t())

        simulation_required = True  # assume true; we may chance this in a moment...
        if os.path.exists(self.vtu_file_name):
            vtk_file_checker_reader = VtkDataReader.VtkDataReader(self.vtu_file_name,
                                                                  parameters_container,
                                                                  None)
            simulation_required = not vtk_file_checker_reader.has_simulation_output_data()

        if simulation_required:
            # Run the simulation
            subprocess.run(['mpirun', '-np',
                            self.config_manager.get_num_cores(),
                            self.config_manager.get_ns_solver_exe(),
                            self.mesh_xml_file_name,
                            'conditions.xml']
                           ).check_returncode()

            # Remove the file if it exists before creating a new one. Otherwise, the FIELD_CONVERT_PATH executable
            # will stall, waiting for keyboard input confirming permission to overwrite the existing vtu.
            if os.path.exists(self.vtu_file_name):
                os.remove(self.vtu_file_name)
            subprocess.run(['mpirun', '-np', '1',
                            self.config_manager.get_field_convert_exe(), self.fld_file_name,
                            self.mesh_xml_file_name, self.vtu_file_name]).check_returncode()

        if not os.path.exists(self.vtu_and_xml_file_basename+"_using_points_from_xml.vtu"):
            VtkDataReader.interpolate_vtu_onto_xml_defined_grid(self.vtu_file_name,
                                                                self.mesh_xml_file_name,
                                                                self.vtu_and_xml_file_basename+"_using_points_from_xml.vtu")

        os.chdir(self.initial_working_path)

    def _set_simulation_inflow_parameter(self, t_parameter):
        ActiveLearningUtilities.substitute_text_in_file('conditions.xml', 'y*(10-y)/25',
                                                        'y*(10-y)/25*{}'.format(t_parameter))

    def get_vtu_file_without_extension(self, parameters_container):
        path_to_vtu_file_without_file_extension = self.nektar_data_root_path +\
                                                  self.simulation_subfolder_template.format(parameters_container.get_t(),
                                                                                            parameters_container.get_r()) +\
                                                  '/' +\
                                                  self.vtu_file_name.split('.')[0]
        return path_to_vtu_file_without_file_extension


def generate_parametric_solution(t_parameter, r_parameter):
    base_working_dir = r'/home/chris/WorkData/nektar++/actual/bezier'
    reference_data_subfolder = r'basic'

    config_manager = ConfigManager.ConfigManager()
    simulation_subfolder_template = config_manager.get_mesh_data_folder_template()
    vtu_and_xml_file_basename = config_manager.get_vtu_and_xml_file_basename()

    parameter_container = SimulationParameterManager.SimulationParameterContainer(t_parameter, r_parameter)

    logger = ActiveLearningUtilities.create_logger('SelfTeachingDriver')

    driver = NektarDriver(base_working_dir, reference_data_subfolder,
                          simulation_subfolder_template, vtu_and_xml_file_basename, logger)
    driver.run_simulation(parameter_container)


if __name__ == '__main__':
    t_parameter = 2.0
    # r_parameter = 0.0

    timer = TicToc()
    timer.tic()

    for r_parameter in [-1.975]:
        generate_parametric_solution(t_parameter, r_parameter)

    timer.toc()
