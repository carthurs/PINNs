import os
import distutils.dir_util
import subprocess
import fileinput
import json


def substitute_text_in_file(filename, text_to_replace, replacement_text):
    for line in fileinput.input(filename, inplace=True):
        print(line.replace(text_to_replace, replacement_text), end="")

class NektarDriver(object):
    NUM_CPU_CORES = 'num_cpu_cores'
    NS_SOLVER_PATH = 'navier_stokes_solver_path'
    FIELD_CONVERT_PATH = 'nektar_field_convert_path'
    NEK_MESH_PATH = 'nektar_mesh_gen_path'

    def __init__(self, nektar_data_root_path, reference_data_subfolder, t_parameter, vtu_and_xml_file_basename):
        self.nektar_data_root_path = nektar_data_root_path
        self.reference_data_subfolder = reference_data_subfolder
        self.new_simulation_subfolder = reference_data_subfolder + '_t{}'.format(t_parameter)
        print("Will create and work in folder {}".format(self.new_simulation_subfolder))
        self.t_parameter = t_parameter
        self.vtu_file_name = vtu_and_xml_file_basename + '.vtu'
        self.initial_working_path = os.getcwd()
        self.mesh_xml_file_name = vtu_and_xml_file_basename + '.xml'
        self.fld_file_name = vtu_and_xml_file_basename + '.fld'

        with open('config.json', 'r') as infile:
            self.system_config = json.loads(infile.read())

    def run_simulation(self):
        os.chdir(self.nektar_data_root_path)
        distutils.dir_util.copy_tree(self.reference_data_subfolder, self.new_simulation_subfolder)
        os.chdir(self.new_simulation_subfolder)

        self._generate_mesh(self.t_parameter)

        # Set the peak inflow velocity to be self.t_parameter
        # self._set_simulation_inflow_parameter()

        # Run the simulation
        subprocess.run(['mpirun', '-np', self.system_config[NektarDriver.NUM_CPU_CORES], self.system_config[NektarDriver.NS_SOLVER_PATH], self.mesh_xml_file_name, 'conditions.xml'])

        # Reduce the results into the vtu for paraview etc.
        try:
            os.remove(self.vtu_file_name)
        except FileNotFoundError:
            pass

        subprocess.run(['mpirun', '-np', '1',
                        self.system_config[NektarDriver.FIELD_CONVERT_PATH], self.fld_file_name,
                        self.mesh_xml_file_name, self.vtu_file_name])

        os.chdir(self.initial_working_path)

    def _set_simulation_inflow_parameter(self):
        substitute_text_in_file('conditions.xml', 'y*(10-y)/25', 'y*(10-y)/25*{}'.format(self.t_parameter))

    def get_vtu_file_without_extension_and_parameter(self):
        path_to_vtu_file_without_file_extension = self.nektar_data_root_path + self.new_simulation_subfolder + '/' + self.vtu_file_name.split('.')[0]
        return path_to_vtu_file_without_file_extension, self.t_parameter

    def _generate_mesh(self, domain_shape_parameter):
        substitute_text_in_file('untitled.geo', 'curving_param = 20.0', 'curving_param = {}'.format(domain_shape_parameter))
        subprocess.run(['gmsh', 'untitled.geo', '-2'])
        subprocess.run(
            ['mpirun', '-np', '1', self.system_config[NektarDriver.NEK_MESH_PATH], 'untitled.msh',
             self.mesh_xml_file_name])
        substitute_text_in_file(self.mesh_xml_file_name, 'FIELDS="u"', 'FIELDS="u,v,p"')



if __name__ == '__main__':
    base_working_dir = r'/home/chris/WorkData/nektar++/actual/'
    reference_data_subfolder = r'tube_10mm_diameter_pt2Mesh_correctViscosity'
    t_parameter = 5.0

    driver = NektarDriver(base_working_dir, reference_data_subfolder, t_parameter, 'tube10mm_diameter_1pt0mesh')
    driver.run_simulation()
