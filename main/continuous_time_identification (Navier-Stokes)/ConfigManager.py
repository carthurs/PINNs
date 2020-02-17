import json


class ConfigManager(object):
    NUM_CPU_CORES = 'num_cpu_cores'
    NS_SOLVER_PATH = 'navier_stokes_solver_path'
    FIELD_CONVERT_PATH = 'nektar_field_convert_path'
    NEK_MESH_PATH = 'nektar_mesh_gen_path'
    COMPOSITE_IDS_NOSLIP = 'composite_ids_for_noslip'
    COMPOSITE_IDS_INFLOW = 'composite_ids_for_inflow'
    COMPOSITE_IDS_OUTFLOW = 'composite_ids_for_outflow'
    NEKTAR_DATA_ROOT_PATH = 'nektar_data_root_path'
    MASTER_MODEL_DATA_ROOT_PATH = 'master_model_data_root_path'
    GMSH_PATH = 'gmsh_path'

    def __init__(self):
        with open('config.json', 'r') as infile:
            self.config_data = json.loads(infile.read())

    def get_field_convert_exe(self):
        return self.config_data[ConfigManager.FIELD_CONVERT_PATH]

    def get_num_cores(self):
        return self.config_data[ConfigManager.NUM_CPU_CORES]

    def get_ns_solver_exe(self):
        return self.config_data[ConfigManager.NS_SOLVER_PATH]

    def get_nekmesh_exe(self):
        return self.config_data[ConfigManager.NEK_MESH_PATH]

    def _get_list_of_ints_as_ints(self, json_tag):
        maybe_as_ints = self.config_data[json_tag]
        definitely_as_ints = [int(item) for item in maybe_as_ints]
        return definitely_as_ints

    def get_composite_ids_noslip(self):
        return self._get_list_of_ints_as_ints(ConfigManager.COMPOSITE_IDS_NOSLIP)

    def get_composite_ids_inflow(self):
        return self._get_list_of_ints_as_ints(ConfigManager.COMPOSITE_IDS_INFLOW)

    def get_composite_ids_outflow(self):
        return self._get_list_of_ints_as_ints(ConfigManager.COMPOSITE_IDS_OUTFLOW)

    def get_nektar_data_root_path(self):
        return self.config_data[ConfigManager.NEKTAR_DATA_ROOT_PATH]

    def get_master_model_data_root_path(self):
        return self.config_data[ConfigManager.MASTER_MODEL_DATA_ROOT_PATH]

    def get_gmsh_exe(self):
        return self.config_data[ConfigManager.GMSH_PATH]
