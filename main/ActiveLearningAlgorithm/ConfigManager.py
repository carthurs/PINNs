import json
import os
import ActiveLearningConstants
import pathlib

class UnknownConfigurationInput(Exception):
    pass

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
    PARAVIEW_PYTHON_INTERPRETER = 'paraview_python_interpreter'
    BOUNDARY_ERRORS_FILENAME_TEMPLATE = 'boundary_errors_filename_template'
    MESH_DATA_FOLDER_TEMPLATE = 'mesh_data_folder_template'
    VTU_AND_XML_FILE_BASENAME = 'vtu_and_xml_file_basename'
    ALA_STARTING_INDEX = 'ala_starting_index'
    PROPORTION_OF_TRAINING_DATA_TO_USE = 'proportion_of_training_data_to_use'
    L2_GRID_PLOT_OUTPUT_SUBFOLDER = 'l2_grid_plot_output_subfolder'
    SLACK_PUSH_URL = 'slack_push_url'
    USE_SLACK_NOTIFICATIONS = 'use_slack_notifications'
    TRAINING_STRATEGY = 'training_strategy'
    PARAVIEW_AVAILABLE = 'paraview_available'
    MACHINE_ID = 'machine_id'
    INFLOW_PARAMETER_RANGE_START = 'inflow_parameter_range_start'
    INFLOW_PARAMETER_RANGE_END = 'inflow_parameter_range_end'
    DIAMETER_PARAMETER_RANGE_START = 'diameter_parameter_range_start'
    DIAMETER_PARAMETER_RANGE_END = 'diameter_parameter_range_end'
    PARAMETER_SPACE_POINT_SPACING = 'parameter_space_point_spacing'
    NUMBER_OF_HIDDEN_LAYERS = 'number_of_hidden_layers'
    NETWORK_WIDTH = 'network_width'
    NAVIER_STOKES_LOSS_SCALING = 'navier_stokes_loss_scaling'
    PRESSURE_NODE_LOSS_SCALING = 'pressure_node_loss_scaling'
    CUSTOM_CURVATURE_REFINEMENT_ENABLED = 'custom_curvature_refinement_enabled'


    def __init__(self, config_root=os.getcwd()):
        # Load default settings first
        with open(config_root + '/config.json.default', 'r') as infile:
            self.config_data = json.loads(infile.read())

        # Override defaults with the user's machine-specific settings from config.json
        with open(config_root + '/config.json', 'r') as infile:
            custom_config_data = json.loads(infile.read())

        self.config_data.update(custom_config_data)


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

    def get_paraview_python_interpreter(self):
        return self.config_data[ConfigManager.PARAVIEW_PYTHON_INTERPRETER]

    def get_boundary_errors_filename_template(self):
        return self.config_data[ConfigManager.BOUNDARY_ERRORS_FILENAME_TEMPLATE]

    def get_mesh_data_folder_template(self):
        return self.config_data[ConfigManager.MESH_DATA_FOLDER_TEMPLATE]

    def get_vtu_and_xml_file_basename(self):
        return self.config_data[ConfigManager.VTU_AND_XML_FILE_BASENAME]

    def get_ala_starting_index(self):
        return int(self.config_data[ConfigManager.ALA_STARTING_INDEX])

    def get_proportion_of_training_data_to_use(self):
        return float(self.config_data[ConfigManager.PROPORTION_OF_TRAINING_DATA_TO_USE])

    def get_l2_grid_plot_output_subfolder(self):
        return pathlib.Path(self.config_data[ConfigManager.L2_GRID_PLOT_OUTPUT_SUBFOLDER])

    def get_slack_push_url(self):
        return self.config_data[ConfigManager.SLACK_PUSH_URL]

    def _true_or_false_string_to_bool(self, lowercase_string):
        if lowercase_string == 'true':
            return True
        elif lowercase_string == 'false':
            return False
        else:
            raise UnknownConfigurationInput

    def slack_integration_enabled(self):
        use_slack_string = self.config_data[ConfigManager.USE_SLACK_NOTIFICATIONS].lower()
        return self._true_or_false_string_to_bool(use_slack_string)

    def paraview_available(self):
        paraview_available = self.config_data[ConfigManager.PARAVIEW_AVAILABLE].lower()
        return self._true_or_false_string_to_bool(paraview_available)

    def get_training_strategy(self):
        strategy = self.config_data[ConfigManager.TRAINING_STRATEGY]
        if strategy == 'active':
            return ActiveLearningConstants.TrainingStrategies.ACTIVE
        elif strategy == 'random':
            return ActiveLearningConstants.TrainingStrategies.RANDOM
        else:
            raise UnknownConfigurationInput

    def get_machine_id(self):
        return self.config_data[ConfigManager.MACHINE_ID]

    def get_inflow_parameter_range_start(self):
        return float(self.config_data[ConfigManager.INFLOW_PARAMETER_RANGE_START])

    def get_inflow_parameter_range_end(self):
        return float(self.config_data[ConfigManager.INFLOW_PARAMETER_RANGE_END])

    def get_diameter_parameter_range_start(self):
        return float(self.config_data[ConfigManager.DIAMETER_PARAMETER_RANGE_START])

    def get_diameter_parameter_range_end(self):
        return float(self.config_data[ConfigManager.DIAMETER_PARAMETER_RANGE_END])

    def get_parameter_space_point_spacing(self):
        config_point_spacing = self.config_data[ConfigManager.PARAMETER_SPACE_POINT_SPACING]
        points_per_unit_parameter_interval = 1.0/float(config_point_spacing)

        distance_from_being_an_integer = abs(round(points_per_unit_parameter_interval) - points_per_unit_parameter_interval)
        if distance_from_being_an_integer > 1e-3:
            raise RuntimeError("Config parameter value {} should be the reciprocal of an integer."
                               " It was {} (error {}).".format(
                                    ConfigManager.PARAMETER_SPACE_POINT_SPACING,
                config_point_spacing,
                distance_from_being_an_integer))

        return float(config_point_spacing)

    def get_number_of_hidden_layers(self):
        return int(self.config_data[ConfigManager.NUMBER_OF_HIDDEN_LAYERS])

    def get_network_width(self):
        return int(self.config_data[ConfigManager.NETWORK_WIDTH])

    def get_navier_stokes_loss_scaling(self):
        return float(self.config_data[ConfigManager.NAVIER_STOKES_LOSS_SCALING])

    def get_pressure_node_loss_scaling(self):
        return float(self.config_data[ConfigManager.PRESSURE_NODE_LOSS_SCALING])

    def custom_curvature_refinement_enabled(self):
        use_custom_curvature_refinement = self.config_data[ConfigManager.CUSTOM_CURVATURE_REFINEMENT_ENABLED].lower()
        return self._true_or_false_string_to_bool(use_custom_curvature_refinement)
