import vtk
from vtk.util import numpy_support
import numpy as np
import scipy.interpolate
import NavierStokes


class VtkDataReader(object):
    def __init__(self, filename):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()

        data_reader = reader.GetOutput()
        self.scalar_point_data = data_reader.GetPointData()
        self.points_vtk = data_reader.GetPoints().GetData()

    def get_data_reader_as_numpy_array(self, array_name):
        array_data = self.scalar_point_data.GetScalars(array_name)
        return numpy_support.vtk_to_numpy(array_data)

    def get_point_coordinates(self):
        return numpy_support.vtk_to_numpy(self.points_vtk)

    def mimic_pinns_input_data(self):
        return_data = dict()

        raw_pressure_data = self.get_data_reader_as_numpy_array('p')
        pressure_data = np.expand_dims(raw_pressure_data, axis=1)  # because the NS code expects a time axis too
        return_data['p_star'] = pressure_data

        time_data = np.zeros(shape=(1, 1))
        return_data['t'] = time_data

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



        return return_data



if __name__ == '__main__':
    # Just a test / usage example - no actual functionality
    my_reader = VtkDataReader(r'E:\dev\PINNs\PINNs\main\Data\tube_10mm_diameter_baselineInflow\tube10mm_diameter_pt05mesh.vtu')
    print(my_reader.get_data_reader_as_numpy_array('u'))
    print(my_reader.get_point_coordinates()[:,0:2])
    pinns_input_format_data = my_reader.mimic_pinns_input_data()

    NavierStokes.plot_solution(pinns_input_format_data['X_star'], pinns_input_format_data['U_star'][:, 0, 0], 1,
                               "True Velocity U", range=[-0.4, 1.0])
    NavierStokes.plot_solution(pinns_input_format_data['X_star'], pinns_input_format_data['U_star'][:, 1, 0], 2,
                               "True Velocity V", range=[-0.23, 0.185])
