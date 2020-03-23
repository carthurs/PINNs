import os
import distutils.dir_util
import subprocess
import fileinput
import VtkDataReader
import ConfigManager
import SimulationParameterManager


def substitute_text_in_file(filename, text_to_replace, replacement_text):
    for line in fileinput.input(filename, inplace=True):
        print(line.replace(text_to_replace, replacement_text), end="")


# Checks if nektar mesh xml file is compressed
def is_compressed(filename):
    with open(filename, 'r') as infile:
        for line in infile:
            if "COMPRESSED" in line:
                return True
    return False


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
        starting_dir = os.getcwd()

        os.chdir(self.nektar_data_root_path)
        simulation_subfolder = self.simulation_subfolder_template.format(parameters_container.get_t(),
                                                                         parameters_container.get_r())

        distutils.dir_util.copy_tree(self.reference_data_subfolder, simulation_subfolder)
        os.chdir(simulation_subfolder)
        self._generate_mesh(parameters_container.get_r())
        if generate_vtu_without_solution_data and not os.path.exists(self.vtu_file_name):
            subprocess.run(['mpirun', '-np', '1',
                            self.config_manager.get_field_convert_exe(),
                            self.mesh_xml_file_name, self.vtu_file_name]).check_returncode()

        os.chdir(starting_dir)

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

            VtkDataReader.interpolate_vtu_onto_xml_defined_grid(self.vtu_file_name,
                                                                self.mesh_xml_file_name,
                                                                self.vtu_and_xml_file_basename+"_using_points_from_xml.vtu")

        os.chdir(self.initial_working_path)

    def _set_simulation_inflow_parameter(self, t_parameter):
        substitute_text_in_file('conditions.xml', 'y*(10-y)/25', 'y*(10-y)/25*{}'.format(t_parameter))

    def get_vtu_file_without_extension(self, parameters_container):
        path_to_vtu_file_without_file_extension = self.nektar_data_root_path +\
                                                  self.simulation_subfolder_template.format(parameters_container.get_t(),
                                                                                            parameters_container.get_r()) +\
                                                  '/' +\
                                                  self.vtu_file_name.split('.')[0]
        return path_to_vtu_file_without_file_extension

    def _generate_mesh(self, domain_shape_parameter):
        if os.path.exists(self.mesh_xml_file_name) and not is_compressed(self.mesh_xml_file_name):
            self.logger.info('Not generating mesh xml file {} because it exists already.'.format(
                os.path.join(os.getcwd(), self.mesh_xml_file_name)))
        else:
            substitute_text_in_file('untitled.geo', 'curving_param = 20.0', 'curving_param = {}'.format(domain_shape_parameter))
            meshing_process_outcome = subprocess.run([self.config_manager.get_gmsh_exe(), 'untitled.geo', '-2'])
            self.logger.info('Return code of gmsh call was {}.'.format(meshing_process_outcome.returncode))

            subprocess.run(
                ['mpirun', '-np', '1', self.config_manager.get_nekmesh_exe(), 'untitled.msh',
                 self.mesh_xml_file_name + ':xml:uncompress']).check_returncode()

            substitute_text_in_file(self.mesh_xml_file_name, 'FIELDS="u"', 'FIELDS="u,v,p"')



if __name__ == '__main__':
    base_working_dir = r'/home/chris/WorkData/nektar++/actual/'
    reference_data_subfolder = r'tube_10mm_diameter_pt2Mesh_correctViscosity'
    ref_data_subfolder_template = reference_data_subfolder + r'_{}'
    t_parameter = 5.0
    r_parameter = 4.0

    parameter_container = SimulationParameterManager.SimulationParameterContainer(t_parameter, r_parameter)

    driver = NektarDriver(base_working_dir, reference_data_subfolder,
                          ref_data_subfolder_template, 'tube10mm_diameter_1pt0mesh')
    driver.run_simulation(parameter_container)
