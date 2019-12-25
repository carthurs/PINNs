import vtk
from vtk.util import numpy_support


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


if __name__ == '__main__':
    # Just a test / usage example - no actual functionality
    my_reader = VtkDataReader(r'E:\dev\PINNs\PINNs\main\Data\tube_10mm_diameter_baselineInflow\tube10mm_diameter_pt05mesh.vtu')
    print(my_reader.get_data_reader_as_numpy_array('u'))
    print(my_reader.get_point_coordinates())
