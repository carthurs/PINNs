import vtk
from vtk.util import numpy_support
import numpy as np
import scipy.interpolate
import NavierStokes
import hashlib
import pickle
import os
import plotly.graph_objects as go
import NektarXmlHandler
import ConfigManager
import BoundaryConditionCodes as BC

def md5hash_file(filename):
    hasher = hashlib.md5()
    with open(filename, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class VtkDataReader(object):
    output_data_version = 1
    NEKTAR_FIX_FACTOR = 0.00106  # workaround bug with Nektar++, which ignores density input parameter rho
    MODE_STRUCTURED = 0
    MODE_UNSTRUCTURED = 1

    def __init__(self, full_filename, time_value_in, data_cache_path):
        self.time_value = np.expand_dims(np.array([time_value_in]), axis=1)

        self.filename_without_extension, ignored_extension = os.path.splitext(full_filename)

        file_md5hash = md5hash_file(full_filename)
        cached_output_filename = "{}.v{}.pickle".format(file_md5hash, VtkDataReader.output_data_version)
        if data_cache_path is not None:
            self.cached_output_filename_fullpath = os.path.join(data_cache_path, cached_output_filename)
        else:
            self.cached_output_filename_fullpath = None

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(full_filename)
        reader.Update()

        self.unstructured_grid = reader.GetOutput()
        self.scalar_point_data = self.unstructured_grid.GetPointData()
        self.points_vtk = self.unstructured_grid.GetPoints().GetData()

    @classmethod
    def from_single_data_file(cls, filename: str):
        return cls(filename, 1.0)

    @classmethod
    def from_single_data_file_with_time_index(cls, filename: str, time_value: float, data_cache_path):
        return cls(filename, time_value, data_cache_path)

    def get_read_data_as_numpy_array(self, array_name):
        array_data = self.scalar_point_data.GetScalars(array_name)
        if array_data is None:
            return_value = None
        else:
            return_value = numpy_support.vtk_to_numpy(array_data)

        return return_value

    def get_point_coordinates(self):
        return numpy_support.vtk_to_numpy(self.points_vtk)

    def _get_pressure_data(self):
        raw_pressure_data = self.get_read_data_as_numpy_array('p')
        if raw_pressure_data is not None:
            pressure_data = np.expand_dims(raw_pressure_data, axis=1)  # because the NS code expects a time axis too
            pressure_data = pressure_data * VtkDataReader.NEKTAR_FIX_FACTOR
        else:
            pressure_data = None
        return pressure_data

    def _get_coordinate_data(self):
        raw_coordinates_data = self.get_point_coordinates()
        return raw_coordinates_data[:, 0:2]

    def get_data_by_mode(self, mode):
        if mode == VtkDataReader.MODE_UNSTRUCTURED:
            return self.get_unstructured_mesh_format_input_data()
        elif mode == VtkDataReader.MODE_STRUCTURED:
            return self.get_pinns_format_input_data()
        else:
            raise RuntimeError('Unknown parameter passed: mode')

    def has_simulation_output_data(self):
        return (self.get_read_data_as_numpy_array('u') is not None)

    def _get_bc_codes(self, num_nodes_in_whole_mesh):
        config_manager = ConfigManager.ConfigManager()

        with open(self.filename_without_extension + '.xml', 'rb') as xml_infile:
            xml_reader = NektarXmlHandler.NektarXmlHandler(xml_infile)
        bc_codes = np.zeros((num_nodes_in_whole_mesh,), dtype=np.float32)

        # set the noslip codes:
        noslip_composite_ids = config_manager.get_composite_ids_noslip()
        noslip_node_ids = xml_reader.get_nodes_in_composite(noslip_composite_ids)
        for node in noslip_node_ids:
            bc_codes[node] = BC.Codes.NOSLIP

        # Set the inflow codes:
        inflow_composite_ids = config_manager.get_composite_ids_inflow()
        inflow_node_ids = xml_reader.get_nodes_in_composite(inflow_composite_ids)
        for node in inflow_node_ids:
            bc_codes[node] = BC.Codes.INFLOW

        # Set the outflow codes:
        outflow_composite_ids = config_manager.get_composite_ids_outflow()
        outflow_node_ids = xml_reader.get_nodes_in_composite(outflow_composite_ids)
        for node in outflow_node_ids:
            bc_codes[node] = BC.Codes.OUTFLOW

        return bc_codes

    def get_unstructured_mesh_format_input_data(self):
        return_data = dict()
        return_data['X_star'] = self._get_coordinate_data()
        return_data['p_star'] = self._get_pressure_data()
        return_data['t'] = self.time_value

        raw_velocity_data_x_component = self.get_read_data_as_numpy_array('u')
        raw_velocity_data_y_component = self.get_read_data_as_numpy_array('v')

        if raw_velocity_data_y_component is None or raw_velocity_data_x_component is None:
            return_data['U_star'] = None
        else:
            velocity_vector_data = np.column_stack((raw_velocity_data_x_component, raw_velocity_data_y_component))
            velocity_data = np.expand_dims(velocity_vector_data, axis=2)  # because the NS code expects a time axis too
            return_data['U_star'] = velocity_data

        num_nodes_in_mesh = return_data['X_star'].shape[0]
        return_data['bc_codes'] = self._get_bc_codes(num_nodes_in_mesh)

        return return_data

    def get_pinns_format_input_data(self):
        # see if we have already interpolated and saved this data. If not, we had better interpolate it now.
        try:
            if self.cached_output_filename_fullpath is not None:
                with open(self.cached_output_filename_fullpath, 'rb') as infile:
                    return_data = pickle.load(infile)
                return return_data
        except FileNotFoundError as e:
            return_data = dict()

        return_data['t'] = self.time_value

        irregular_mesh_coordinates = self._get_coordinate_data()

        x_min = np.min(irregular_mesh_coordinates[:, 0])
        x_max = np.max(irregular_mesh_coordinates[:, 0])
        y_min = np.min(irregular_mesh_coordinates[:, 1])
        y_max = np.max(irregular_mesh_coordinates[:, 1])

        xi = np.linspace(x_min, x_max, 100)
        yi = np.linspace(y_min, y_max, 50)

        xi2, yi2 = np.meshgrid(xi, yi)
        xi2_vector = np.expand_dims(np.reshape(xi2, (100 * 50)), axis=1)
        yi2_vector = np.expand_dims(np.reshape(yi2, (100 * 50)), axis=1)
        full_grid_data_coordinates = np.concatenate((xi2_vector, yi2_vector), axis=1)

        press_data = self._get_pressure_data()
        if press_data is None:
            return_data['p_star'] = None
        else:
            return_data['p_star'] = scipy.interpolate.griddata(irregular_mesh_coordinates, press_data,
                                                               full_grid_data_coordinates,
                                                               method='cubic', fill_value=0.0)

        return_data['X_star'] = full_grid_data_coordinates


        raw_velocity_data_x_component = self.get_read_data_as_numpy_array('u')
        raw_velocity_data_y_component = self.get_read_data_as_numpy_array('v')

        if raw_velocity_data_x_component is None or raw_velocity_data_y_component is None:
            return_data['U_star'] = None
        else:
            raw_velocity_data_x_component = scipy.interpolate.griddata(irregular_mesh_coordinates, raw_velocity_data_x_component,
                                                               (np.reshape(xi2, (100 * 50)), np.reshape(yi2, (100 * 50))),
                                                               method='cubic', fill_value=0.0)

            raw_velocity_data_y_component = scipy.interpolate.griddata(irregular_mesh_coordinates, raw_velocity_data_y_component,
                                                               (np.reshape(xi2, (100 * 50)), np.reshape(yi2, (100 * 50))),
                                                               method='cubic', fill_value=0.0)

            velocity_vector_data = np.column_stack((raw_velocity_data_x_component, raw_velocity_data_y_component))
            velocity_data = np.expand_dims(velocity_vector_data, axis=2)  # because the NS code expects a time axis too
            return_data['U_star'] = velocity_data

        if self.cached_output_filename_fullpath is not None:
            with open(self.cached_output_filename_fullpath, 'wb') as cachefile:
                pickle.dump(return_data, cachefile)

        return return_data

    # TODO refactor this into its own class for plotting...
    def plotly_plot_mesh(self):
        raw_coordinates_data = self.get_point_coordinates()
        x_coords = raw_coordinates_data[:, 0]
        y_coords = raw_coordinates_data[:, 1]
        z_coords_upperplane = raw_coordinates_data[:, 2] + 2.0

        number_of_triangles = self.unstructured_grid.GetNumberOfCells()
        triangle_firstnodes = [0] * number_of_triangles
        triangle_secondnodes = [0] * number_of_triangles
        triangle_thirdnodes = [0] * number_of_triangles

        for triangle_idx in range(number_of_triangles):
            triangle_nodes = self.unstructured_grid.GetCell(triangle_idx).GetPointIds()
            triangle_firstnodes[triangle_idx] = triangle_nodes.GetId(0)
            triangle_secondnodes[triangle_idx] = triangle_nodes.GetId(1)
            triangle_thirdnodes[triangle_idx] = triangle_nodes.GetId(2)

        pressure_plot = go.Mesh3d(x=x_coords, y=y_coords, z=z_coords_upperplane,
                                  i=triangle_firstnodes, j=triangle_secondnodes, k=triangle_thirdnodes,
                                  intensity=self.get_read_data_as_numpy_array('p'),
                                  colorscale='Picnic',
                                  colorbar_title='Pressure')


        mydata = self.get_pinns_format_input_data()

        z_coords_lowerplane = mydata['X_star'][:, 0]*0.0 - 2.0
        velocity_plot = go.Cone(x=mydata['X_star'][:, 0], y=mydata['X_star'][:, 1], z=z_coords_lowerplane,
                                u=mydata['U_star'][:, 0, 0],
                                v=mydata['U_star'][:, 1, 0],
                                w=mydata['U_star'][:, 1, 0] * 0.0,
                                sizeref=10.0,
                                colorscale='RdBu',
                                colorbar=dict(x=0.0),
                                colorbar_title='Velocity Magnitude')

        fig = go.Figure(data=[velocity_plot, pressure_plot])
        fig.show()


class MultipleFileReader(object):
    # mode should be either 'unstructured' (raw mesh node data), or 'structured' (interpolated on to a regular grid)
    def __init__(self, cached_data_path_in, mode='structured'):
        self.gathered_data = None
        self.cached_data_path = cached_data_path_in
        if mode == 'structured':
            self.mode = VtkDataReader.MODE_STRUCTURED
        elif mode == 'unstructured':
            self.mode = VtkDataReader.MODE_UNSTRUCTURED
        else:
            raise RuntimeError('Unknown value for parameter: mode.')
        self.file_names_by_parameter_values = dict()

    def add_file_name(self, file_name, parameter_value):
        data_for_this_file = VtkDataReader.from_single_data_file_with_time_index(file_name, parameter_value,
                                                                                 self.cached_data_path) \
                                                                                .get_data_by_mode(self.mode)

        self.file_names_by_parameter_values[parameter_value] = file_name

        # Reformat the data so that we have arrays for x, y, u, v, p, and t which are 1D and all of the same length
        # num_parameter_slices*T, and such that the i-th entry of all arrays correspond to one another.
        # This is the correct format for streaming the data through the tensorflow placeholders during training.
        data_for_this_file_streaming_format = dict()

        X_star = data_for_this_file['X_star']  # N x 2
        t_star = data_for_this_file['t']  # num_parameter_slices x 1

        N = X_star.shape[0]
        num_parameter_slices = t_star.shape[0]

        TT = np.tile(t_star, (1, N)).T  # N x num_parameter_slices
        data_for_this_file_streaming_format['t'] = TT.flatten()[:, None]  # N*num_parameter_slices x 1

        XX = np.tile(X_star[:, 0:1], (1, num_parameter_slices))  # N x num_parameter_slices
        x = XX.flatten()[:, None]  # N*num_parameter_slices x 1
        data_for_this_file_streaming_format['x'] = x

        YY = np.tile(X_star[:, 1:2], (1, num_parameter_slices))  # N x num_parameter_slices
        y = YY.flatten()[:, None]  # N*num_parameter_slices x 1
        data_for_this_file_streaming_format['y'] = y

        U_star = data_for_this_file['U_star']  # num_parameter_slices x 2 x T
        UU = U_star[:, 0, :]  # num_parameter_slices x T
        data_for_this_file_streaming_format['u'] = UU.flatten()[:, None]  # num_parameter_slices*T x 1

        VV = U_star[:, 1, :]  # num_parameter_slices x T
        data_for_this_file_streaming_format['v'] = VV.flatten()[:, None]  # num_parameter_slices*T x 1

        P_star = data_for_this_file['p_star']  # num_parameter_slices x T
        PP = P_star  # num_parameter_slices x T
        data_for_this_file_streaming_format['p'] = PP.flatten()[:,None]  # num_parameter_slices*T x 1

        data_for_this_file_streaming_format['bc_codes'] = data_for_this_file['bc_codes']

        if self.gathered_data is None:
            self.gathered_data = data_for_this_file_streaming_format
        else:
            # Concatenate with those data arrays which need extending to account for the new data:
            for key in data_for_this_file_streaming_format:
                self.gathered_data[key] = np.concatenate((self.gathered_data[key],
                                                          data_for_this_file_streaming_format[key]), 0)

    def add_point_for_navier_stokes_loss_only(self, point):
        if self.gathered_data is None:
            raise RuntimeError("Please add some data with add_file_name before calling this function.")
        else:
            for key in self.gathered_data:
                if key in MultipleFileReader.key_to_concatenation_axis_map:
                    if key == 't':
                        point_for_navier_stokes_loss_only = self.gathered_data[key] * 0 + point
                        self.gathered_data[key] = np.concatenate((self.gathered_data[key], point_for_navier_stokes_loss_only),
                                                                 MultipleFileReader.key_to_concatenation_axis_map[key])
                    else:
                        dummy_data_to_be_ignored = self.gathered_data[key] * 0 - 1.0  # -1 to indicate  that it's dummy data
                        self.gathered_data[key] = np.concatenate((self.gathered_data[key], dummy_data_to_be_ignored),
                                                                 MultipleFileReader.key_to_concatenation_axis_map[key])
        return self

    def get_training_data(self):
        return self.gathered_data

    def get_test_data(self, t_parameter_for_test):
        # Might want to switch this name out for  something passed in instead. Currently we're just evaluating on
        # a full space-time slice from which we've sampled training data (which is why its file name is in the
        # dictionary file_names_by_parameter_values).
        file_name = self.file_names_by_parameter_values[t_parameter_for_test]

        raw_data_for_test = VtkDataReader.from_single_data_file_with_time_index(file_name,
                                                                            t_parameter_for_test,
                                                                            self.cached_data_path) \
                                                                        .get_data_by_mode(self.mode)

        X_star = raw_data_for_test['X_star']  # N x 2

        test_data = dict()
        test_data['X_star'] = X_star
        test_data['x'] = X_star[:, 0:1]
        test_data['y'] = X_star[:, 1:2]

        N = len(test_data['y'])
        test_data['t'] = np.ones((N, 1)) * t_parameter_for_test

        U_star = raw_data_for_test['U_star']  # num_parameter_slices x 2 x T
        test_data['u'] = U_star[:, 0]
        test_data['v'] = U_star[:, 1]

        test_data['p'] = raw_data_for_test['p_star']

        test_data['bc_codes'] = raw_data_for_test['bc_codes']

        return test_data


if __name__ == '__main__':
    # Just a test / usage example - no actual functionality
    # my_reader = VtkDataReader(r'E:\dev\PINNs\PINNs\main\Data\tube_10mm_diameter_baselineInflow\tube_10mm_diameter_pt2Mesh_correctViscosity\tube10mm_diameter_pt05mesh.vtu', os.getcwd())
    my_reader = VtkDataReader(r'/home/chris/WorkData/nektar++/actual/bezier/basic_t3.0/tube_bezier_1pt0mesh.vtu', 1.0, r'/home/chris/WorkData/nektar++/actual/bezier/master_data/')
    print(my_reader.get_read_data_as_numpy_array('u'))
    print(my_reader.get_point_coordinates()[:,0:2])

    data = my_reader.get_pinns_format_input_data()

    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 2

    my_reader.plotly_plot_mesh()

    # NavierStokes.plot_solution(pinns_input_format_data['X_star'], data['U_star'][:, 0, 0], 1,
    #                            "True Velocity U", colour_range=[0.0, 1.0])
    # NavierStokes.plot_solution(pinns_input_format_data['X_star'], data['U_star'][:, 1, 0], 2,
    #                            "True Velocity V", colour_range=[-0.00009, 0.00017])
