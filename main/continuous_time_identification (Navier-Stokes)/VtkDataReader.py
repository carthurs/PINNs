import vtk
from vtk.util import numpy_support
import numpy as np
import scipy.interpolate
import NavierStokes
import hashlib
import pickle

def md5hash_file(filename):
    hasher = hashlib.md5()
    with open(filename, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class VtkDataReader(object):
    output_data_version = 1
    def __init__(self, filename, time_value_in):
        self.time_value = np.expand_dims(np.array([time_value_in]), axis=1)

        file_md5hash = md5hash_file(filename)
        self.cached_output_filename = "{}.v{}.pickle".format(file_md5hash, VtkDataReader.output_data_version)

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()

        data_reader = reader.GetOutput()
        self.scalar_point_data = data_reader.GetPointData()
        self.points_vtk = data_reader.GetPoints().GetData()

    @classmethod
    def from_single_data_file(cls, filename: str):
        return cls(filename, 1.0)

    @classmethod
    def from_single_data_file_with_time_index(cls, filename: str, time_value: float):
        return cls(filename, time_value)

    def get_data_reader_as_numpy_array(self, array_name):
        array_data = self.scalar_point_data.GetScalars(array_name)
        return numpy_support.vtk_to_numpy(array_data)

    def get_point_coordinates(self):
        return numpy_support.vtk_to_numpy(self.points_vtk)

    def get_pinns_format_input_data(self):
        # see if we have already interpolated and saved this data. If not, we had better interpolate it now.
        try:
            with open(self.cached_output_filename, 'rb') as infile:
                return_data = pickle.load(infile)
            return return_data
        except FileNotFoundError as e:
            return_data = dict()

        raw_pressure_data = self.get_data_reader_as_numpy_array('p')
        pressure_data = np.expand_dims(raw_pressure_data, axis=1)  # because the NS code expects a time axis too
        pressure_data = pressure_data * 0.00106  # fix screwup in Nektar++
        return_data['p_star'] = pressure_data

        return_data['t'] = self.time_value

        raw_coordinates_data = self.get_point_coordinates()
        irregular_mesh_coordinates = raw_coordinates_data[:, 0:2]

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

        return_data['p_star'] = scipy.interpolate.griddata(irregular_mesh_coordinates, return_data['p_star'],
                                                           full_grid_data_coordinates,
                                                           method='cubic')

        return_data['X_star'] = full_grid_data_coordinates


        raw_velocity_data_x_component = self.get_data_reader_as_numpy_array('u')
        raw_velocity_data_y_component = self.get_data_reader_as_numpy_array('v')

        raw_velocity_data_x_component = scipy.interpolate.griddata(irregular_mesh_coordinates, raw_velocity_data_x_component,
                                                           (np.reshape(xi2, (100 * 50)), np.reshape(yi2, (100 * 50))),
                                                           method='cubic')

        raw_velocity_data_y_component = scipy.interpolate.griddata(irregular_mesh_coordinates, raw_velocity_data_y_component,
                                                           (np.reshape(xi2, (100 * 50)), np.reshape(yi2, (100 * 50))),
                                                           method='cubic')

        velocity_vector_data = np.column_stack((raw_velocity_data_x_component, raw_velocity_data_y_component))
        velocity_data = np.expand_dims(velocity_vector_data, axis=2)  # because the NS code expects a time axis too
        return_data['U_star'] = velocity_data

        with open(self.cached_output_filename, 'wb') as cachefile:
            pickle.dump(return_data, cachefile)

        return return_data


class MultipleFileReader(object):
    # Static variables
    key_to_concatenation_axis_map = {'U_star': 2, 'p_star': 1, 't': 0}

    def __init__(self):
        self.gathered_data = None

    def add_file_name(self, file_name, parameter_value):
        data_for_this_file = VtkDataReader.from_single_data_file_with_time_index(file_name, parameter_value) \
                                                                                    .get_pinns_format_input_data()
        if self.gathered_data is not None:
            # Concatenate with those data arrays which need extending to account for the new data:
            for key in data_for_this_file:
                if key in MultipleFileReader.key_to_concatenation_axis_map:
                    self.gathered_data[key] = np.concatenate((self.gathered_data[key], data_for_this_file[key]),
                                                             MultipleFileReader.key_to_concatenation_axis_map[key])
        else:
            self.gathered_data = data_for_this_file

    def get_pinns_format_input_data(self):
        return self.gathered_data




if __name__ == '__main__':
    # Just a test / usage example - no actual functionality
    my_reader = VtkDataReader(r'E:\dev\PINNs\PINNs\main\Data\tube_10mm_diameter_baselineInflow\tube_10mm_diameter_pt2Mesh_correctViscosity\tube10mm_diameter_pt05mesh.vtu')
    print(my_reader.get_data_reader_as_numpy_array('u'))
    print(my_reader.get_point_coordinates()[:,0:2])
    pinns_input_format_data = my_reader.get_pinns_format_input_data()

    NavierStokes.plot_solution(pinns_input_format_data['X_star'], pinns_input_format_data['U_star'][:, 0, 0], 1,
                               "True Velocity U", colour_range=[0.0, 1.0])
    NavierStokes.plot_solution(pinns_input_format_data['X_star'], pinns_input_format_data['U_star'][:, 1, 0], 2,
                               "True Velocity V", colour_range=[-0.00009, 0.00017])
