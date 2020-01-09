import os
import distutils.dir_util
import subprocess
import fileinput


class NektarDriver(object):
    def __init__(self, nektar_data_root_path, reference_data_subfolder, t_parameter, vtu_file_name):
        self.nektar_data_root_path = nektar_data_root_path
        self.reference_data_subfolder = reference_data_subfolder
        self.new_simulation_subfolder = reference_data_subfolder + '_t{}'.format(t_parameter)
        print("Will create and work in folder {}".format(self.new_simulation_subfolder))
        self.t_parameter = t_parameter
        self.vtu_file_name = vtu_file_name
        self.initial_working_path = os.getcwd()

    def run_simulation(self):
        os.chdir(self.nektar_data_root_path)
        distutils.dir_util.copy_tree(self.reference_data_subfolder, self.new_simulation_subfolder)
        os.chdir(self.new_simulation_subfolder)

        # Set the peak inflow velocity to be self.t_parameter
        self._set_simulation_inflow_parameter()

        # Run the simulation
        subprocess.run(['mpirun', '-np', '6', '/home/chris/workspace/nektapp/nektar++/build/dist/bin/IncNavierStokesSolver', 'tube10mm_diameter_pt05mesh.xml', 'conditions.xml'])

        # Reduce the results into the vtu for paraview etc.
        os.remove(self.vtu_file_name)
        subprocess.run(['mpirun', '-np', '1',
                        '/home/chris/workspace/nektapp/nektar++/build/dist/bin/FieldConvert', 'tube10mm_diameter_pt05mesh.fld',
                        'tube10mm_diameter_pt05mesh.xml', self.vtu_file_name])

        os.chdir(self.initial_working_path)

    def _set_simulation_inflow_parameter(self):
        for line in fileinput.input("conditions.xml", inplace=True):
            print(line.replace('y*(10-y)/25', 'y*(10-y)/25*{}'.format(self.t_parameter)), end="")

    def get_vtu_file_without_extension_and_parameter(self):
        path_to_vtu_file_without_file_extension = self.nektar_data_root_path + self.new_simulation_subfolder + '/' + self.vtu_file_name.split('.')[0]
        return path_to_vtu_file_without_file_extension, self.t_parameter


if __name__ == '__main__':
    base_working_dir = r'/home/chris/WorkData/nektar++/actual/'
    reference_data_subfolder = r'tube_10mm_diameter_pt2Mesh_correctViscosity'
    t_parameter = 5.0

    driver = NektarDriver(base_working_dir, reference_data_subfolder, t_parameter)
    driver.run_simulation()
