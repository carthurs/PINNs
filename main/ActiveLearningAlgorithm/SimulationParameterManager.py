import numpy as np
import logging
import hashlib
import random
import ActiveLearningUtilities

class SimulationParameterContainer(object):
    # Careful with this class. It should not contain any lists, dicts or other mutable types, or else
    # it will not be usable as a dictionary key. If you get into that situation, override the default
    # __hash__() and __eq__() methods.
    def __init__(self, t, r):
        self.__t = t
        self.__r = r
        content_based_id_string = 'r={},t={}'.format(r, t)
        self.content_derived_id = int(hashlib.sha256(content_based_id_string.encode('utf-8')).hexdigest(), 16)

    def get_t(self):
        return self.__t

    def get_r(self):
        return self.__r

    def near(self, other, tolerance):
        if abs(other.get_r() - self.__r) < tolerance and abs(other.get_t() - self.__t) < tolerance:
            return True
        else:
            return False

    def log(self):
        info_string = 'Will get data for parameter values t={}, r={}'.format(self.__t, self.__r)
        logger = logging.getLogger('SelfTeachingDriver')
        logger.info(info_string)

    def __str__(self):
        return 't={}, r={}'.format(self.__t, self.__r)

    def __hash__(self):
        return self.content_derived_id

    def __eq__(self, other):
        if not isinstance(other, SimulationParameterContainer):
            return False
        return self.content_derived_id == other.content_derived_id

    def __lt__(self, other):
        if not isinstance(other, SimulationParameterContainer):
            raise TypeError("cannot compare types {} and {}.".format(type(self), type(other)))
        if self.__t < other.get_t():
            return True
        elif self.__t == other.get_t():
            if self.__r < other.get_r():
                return True
        else:
            return False


class SimulationParameterManager(object):
    def __init__(self, parameter_descriptor_t, parameter_descriptor_r):
        # Let's name t as the inflow parameter, and r as the curvature ("~radius") parameter
        t_parameter_linspace = np.linspace(parameter_descriptor_t['range_start'], parameter_descriptor_t['range_end'], num=parameter_descriptor_t['number_of_points'])
        r_parameter_linspace = np.linspace(parameter_descriptor_r['range_start'], parameter_descriptor_r['range_end'], num=parameter_descriptor_r['number_of_points'])

        self.parameter_dimensionality = 2

        self.__parameter_space = np.meshgrid(t_parameter_linspace, r_parameter_linspace)
        self.flat_t_space = self.__parameter_space[0].flatten()
        self.flat_r_space = self.__parameter_space[1].flatten()

        self.number_of_unused_parameter_points = len(self.flat_r_space)
        self.used_parameter_indices = list()

    def annotate_parameter_set_as_used(self, parameter_container):
        for index, params in enumerate(self.all_parameter_points()):
            if parameter_container == params:
                if index in self.used_parameter_indices:
                    raise RuntimeError('Attempted to mark parameter index {} as used, but it was already marked as such.')
                self.used_parameter_indices.append(index)
                self.number_of_unused_parameter_points -= 1
                if self.number_of_unused_parameter_points < 0:
                    raise RuntimeError('Number of unused parameter points became negative.')
                break

    def get_random_unused_parameter_set(self):
        if self.number_of_unused_parameter_points == 0:
            raise ActiveLearningUtilities.NoAvailableParameters

        random_unused_parameter_index = random.randint(0, self.number_of_unused_parameter_points - 1)  # randint(a, b) generates in the range [a, b] inclusive, hence the -1
        unused_parameter_counter = 0
        for index, param in enumerate(self.all_parameter_points()):
            if index not in self.used_parameter_indices:  # only interested in indexing over the unused parameters
                if unused_parameter_counter == random_unused_parameter_index:
                    self.annotate_parameter_set_as_used(param)
                    return param
                else:
                    unused_parameter_counter += 1

        raise RuntimeError('Failed to find unused parameter set, but some should have been available.')

    def get_initial_parameters(self, strategy):
        all_initial_parameters = list()
        if strategy == 'end_value':
            t_end = self.flat_t_space[-1]
            r_end = self.flat_r_space[-1]
            all_initial_parameters.append(SimulationParameterContainer(t_end, r_end))
        elif strategy == 'corners':  # need to custom-define these here
            all_initial_parameters.append(SimulationParameterContainer(2.0, 2.0))
            all_initial_parameters.append(SimulationParameterContainer(-2.0, 2.0))
            all_initial_parameters.append(SimulationParameterContainer(2.0, -2.0))
            all_initial_parameters.append(SimulationParameterContainer(-2.0, -2.0))
        elif strategy == 'corners_and_centre':  # need to custom-define these here
            all_initial_parameters = list()
            all_initial_parameters.append(SimulationParameterContainer(2.0, 2.0))
            all_initial_parameters.append(SimulationParameterContainer(-2.0, 2.0))
            all_initial_parameters.append(SimulationParameterContainer(2.0, -2.0))
            all_initial_parameters.append(SimulationParameterContainer(-2.0, -2.0))
            all_initial_parameters.append(SimulationParameterContainer(0.0, 0.0))
        else:
            raise RuntimeError('Unknown strategy provided.')

        for params in all_initial_parameters:
            self.annotate_parameter_set_as_used(params)

        return all_initial_parameters

    def get_parameter_space(self):
        return self.__parameter_space

    def all_parameter_points(self):
        for t, r in zip(self.flat_t_space, self.flat_r_space):
            yield SimulationParameterContainer(t, r)

    def get_num_parameter_points(self):
        return len(self.flat_t_space)

    def get_parameter_dimensionality(self):
        return self.parameter_dimensionality
