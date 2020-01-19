import os
import distutils.dir_util
import subprocess
import fileinput
import json
import VtkDataReader


def substitute_text_in_file(filename, text_to_replace, replacement_text):
    for line in fileinput.input(filename, inplace=True):
        print(line.replace(text_to_replace, replacement_text), end="")

class NektarDriver(object):
    NUM_CPU_CORES = 'num_cpu_cores'
    NS_SOLVER_PATH = 'navier_stokes_solver_path'
    FIELD_CONVERT_PATH = 'nektar_field_convert_path'
    NEK_MESH_PATH = 'nektar_mesh_gen_path'

    def __init__(self, nektar_data_root_path, reference_data_subfolder, simulation_subfolder_template, t_parameter,
                 vtu_and_xml_file_basename, logger):
        self.nektar_data_root_path = nektar_data_root_path
        self.reference_data_subfolder = reference_data_subfolder
        self.simulation_subfolder_template = simulation_subfolder_template
        self.t_parameter = t_parameter
        print("Will create and work in folder {}".format(self.simulation_subfolder_template.format(self.t_parameter)))
        self.vtu_file_name = vtu_and_xml_file_basename + '.vtu'
        self.initial_working_path = os.getcwd()
        self.mesh_xml_file_name = vtu_and_xml_file_basename + '.xml'
        self.fld_file_name = vtu_and_xml_file_basename + '.fld'
        self.logger = logger

        with open('config.json', 'r') as infile:
            self.system_config = json.loads(infile.read())

    def generate_vtu_mesh_for_parameter(self, t_param):
        self._prepare_simulation_files(t_param, generate_vtu_without_solution_data=True)


    def _prepare_simulation_files(self, t_param, generate_vtu_without_solution_data=False):
        starting_dir = os.getcwd()

        os.chdir(self.nektar_data_root_path)
        simulation_subfolder = self.simulation_subfolder_template.format(t_param)

        distutils.dir_util.copy_tree(self.reference_data_subfolder, simulation_subfolder)
        os.chdir(simulation_subfolder)
        self._generate_mesh(t_param)
        if generate_vtu_without_solution_data and not os.path.exists(self.vtu_file_name):
            subprocess.run(['mpirun', '-np', '1',
                            self.system_config[NektarDriver.FIELD_CONVERT_PATH],
                            self.mesh_xml_file_name, self.vtu_file_name]).check_returncode()

        os.chdir(starting_dir)

    def run_simulation(self):
        self._prepare_simulation_files(self.t_parameter)
        os.chdir(self.nektar_data_root_path)
        os.chdir(self.simulation_subfolder_template.format(self.t_parameter))

        # Set the peak inflow velocity to be self.t_parameter
        # self._set_simulation_inflow_parameter()

        simulation_required = True  # assume true; we may chance this in a moment...
        if os.path.exists(self.vtu_file_name):
            vtk_file_checker_reader = VtkDataReader.VtkDataReader(self.vtu_file_name,
                                                                  self.t_parameter,
                                                                  None)
            simulation_required = not vtk_file_checker_reader.has_simulation_output_data()

        if simulation_required:
            # Run the simulation
            subprocess.run(['mpirun', '-np', self.system_config[NektarDriver.NUM_CPU_CORES], self.system_config[NektarDriver.NS_SOLVER_PATH], self.mesh_xml_file_name, 'conditions.xml']).check_returncode()

            # Remove the file if it exists before creating a new one. Otherwise, the FIELD_CONVERT_PATH executable
            # will stall, waiting for keyboard input confirming permission to overwrite the existing vtu.
            if os.path.exists(self.vtu_file_name):
                os.remove(self.vtu_file_name)
            subprocess.run(['mpirun', '-np', '1',
                            self.system_config[NektarDriver.FIELD_CONVERT_PATH], self.fld_file_name,
                            self.mesh_xml_file_name, self.vtu_file_name]).check_returncode()

        os.chdir(self.initial_working_path)

    def _set_simulation_inflow_parameter(self):
        substitute_text_in_file('conditions.xml', 'y*(10-y)/25', 'y*(10-y)/25*{}'.format(self.t_parameter))

    def get_vtu_file_without_extension_and_parameter(self):
        path_to_vtu_file_without_file_extension = self.nektar_data_root_path + self.simulation_subfolder_template.format(self.t_parameter) + '/' + self.vtu_file_name.split('.')[0]
        return path_to_vtu_file_without_file_extension, self.t_parameter

    def _generate_mesh(self, domain_shape_parameter):
        if os.path.exists(self.mesh_xml_file_name):
            self.logger.info('Not generating mesh xml file {} because it exists already.'.format(
                os.path.join(os.getcwd(), self.mesh_xml_file_name)))
        else:
            substitute_text_in_file('untitled.geo', 'curving_param = 20.0', 'curving_param = {}'.format(domain_shape_parameter))
            meshing_process_outcome = subprocess.run(['gmsh', 'untitled.geo', '-2'])
            self.logger.info('Return code of gmsh call was {}.'.format(meshing_process_outcome.returncode))

            subprocess.run(
                ['mpirun', '-np', '1', self.system_config[NektarDriver.NEK_MESH_PATH], 'untitled.msh',
                 self.mesh_xml_file_name + ':xml:uncompress']).check_returncode()

            substitute_text_in_file(self.mesh_xml_file_name, 'FIELDS="u"', 'FIELDS="u,v,p"')



if __name__ == '__main__':
    base_working_dir = r'/home/chris/WorkData/nektar++/actual/'
    reference_data_subfolder = r'tube_10mm_diameter_pt2Mesh_correctViscosity'
    ref_data_subfolder_template = reference_data_subfolder + r'_{}'
    t_parameter = 5.0

    driver = NektarDriver(base_working_dir, reference_data_subfolder,
                          ref_data_subfolder_template, t_parameter, 'tube10mm_diameter_1pt0mesh')
    driver.run_simulation()
