# !!!! NOTE !!!!
# this script must be run by the builtin paraview python interpreter
# !!!!!!!!!!!!!!
import paraview.simple as ps
import sys
import json


def run_paraview_integration(input_filename, communication_filename):
    reader = ps.XMLUnstructuredGridReader(FileName=input_filename)
    integrate_filter = ps.IntegrateVariables(reader)
    integrate_filter.UpdatePipeline()

    arrays_to_output = ['u', 'v', 'p']
    integrals = dict()
    non_error_integrals = dict()
    for i in range(100):
        array = integrate_filter.PointData.GetArray(i)
        if array is None:
            break

        array_name = array.GetName()
        print(array_name)
        for array_to_check in arrays_to_output:
            if array_name == '{}_squared'.format(array_to_check):
                non_error_integrals[array_to_check] = array.GetRange()[0]
                break
            elif array_name == '{}_nodewise_squared_error'.format(array_to_check):
                integrals[array_to_check] = array.GetRange()[0]
                break

        if array_name == 'ones':
            integrals['ones'] = array.GetRange()[0]

    print('integrals:', integrals)
    area = integrals['ones']
    area_normalised_integrals = {key: value / area for key, value in integrals.items()}
    print('area_normalised_integrals', area_normalised_integrals)

    integration_output = {'integrals': integrals, 'area_normalised_integrals': area_normalised_integrals, 'non_error_integrals': non_error_integrals}

    with open(communication_filename, 'w') as file:
        json.dump(integration_output, file)
        print("Wrote to file ", communication_filename)


if __name__ =='__main__':

    input_filename = sys.argv[1]
    communication_filename = sys.argv[2]
    run_paraview_integration(input_filename, communication_filename)

