import ActiveLearningUtilities
import subprocess
import ConfigManager
import warnings
import os
import distutils.dir_util

class MeshGenerator(object):
    def __init__(self, mesh_xml_file_name, parameters_container, logger=None):
        self.mesh_xml_file_name = mesh_xml_file_name
        self.parameters_container = parameters_container
        self.logger = logger

        self.config_manager = ConfigManager.ConfigManager()

    # _prepare_simulation_files
    def generate_in_correct_folder(self, vtu_additional_file_name=None):
        starting_dir = os.getcwd()

        nektar_data_root_path = self.config_manager.get_nektar_data_root_path()
        os.chdir(nektar_data_root_path)

        simulation_subfolder_template = self.config_manager.get_mesh_data_folder_template()
        simulation_subfolder = simulation_subfolder_template.format(self.parameters_container.get_t(),
                                                                    self.parameters_container.get_r())

        reference_data_subfolder = r'basic'

        distutils.dir_util.copy_tree(reference_data_subfolder, simulation_subfolder)
        os.chdir(simulation_subfolder)
        self._generate()

        if vtu_additional_file_name is not None and not os.path.exists(vtu_additional_file_name):
            ActiveLearningUtilities.convert_xml_to_vtu(self.mesh_xml_file_name, vtu_additional_file_name,
                                                       config_root=starting_dir)

        os.chdir(starting_dir)

    def _generate(self):
        if os.path.exists(self.mesh_xml_file_name) and not ActiveLearningUtilities.is_compressed(self.mesh_xml_file_name):
            message = 'Not generating mesh xml file {} because it exists already.'.format(
                    os.path.join(os.getcwd(), self.mesh_xml_file_name))

            if self.logger:
                self.logger.info(message)
            else:
                warnings.warn(message)

        else:
            ActiveLearningUtilities.substitute_text_in_file('untitled.geo', 'curving_param = 20.0',
                                    'curving_param = {}'.format(self.parameters_container.get_r()))

            if self.config_manager.custom_curvature_refinement_enabled() and self.parameters_container.get_r() < 0.0:
                # scale the mesh size linearly in r, taking the value 0.2 when r=-9.0, and 1.0 when r=0. Linearly
                # interpolate inbetween. Note this is only hte negative r cases, due to the if-clause we're in.
                fine_mesh_size = 0.2 + (1.0 - abs(self.parameters_container.get_r()/9.0)) * (1.0 - 0.2)
            else:
                fine_mesh_size = 1.0
            ActiveLearningUtilities.substitute_text_in_file('untitled.geo', 'fine_mesh_size = 0.25',
                                                            'fine_mesh_size = {}'.format(
                                                                fine_mesh_size))

            meshing_process_outcome = subprocess.run([self.config_manager.get_gmsh_exe(), 'untitled.geo', '-2'])
            return_message = 'Return code of gmsh call was {}.'.format(meshing_process_outcome.returncode)
            if self.logger:
                self.logger.info(return_message)
            else:
                print(return_message)

            subprocess.run(
                ['mpirun', '-np', '1', self.config_manager.get_nekmesh_exe(), 'untitled.msh',
                 self.mesh_xml_file_name + ':xml:uncompress']).check_returncode()

            ActiveLearningUtilities.substitute_text_in_file(self.mesh_xml_file_name, 'FIELDS="u"', 'FIELDS="u,v,p"')