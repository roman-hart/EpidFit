import inspect
import numpy as np
from typing import Union, Callable
from epidfit.residuals import mse
from epidfit.optimizers import lm
from epidfit.results import Result, ResultsList


class State:
    """ State of compartment at time t """
    def __init__(self, t: float, compartments_names: list, increments_function, data: 'Data' = None, **kwargs):
        self.t, self.compartments_names, self.increments_function = t, compartments_names, increments_function
        self.data = data
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def N(self):
        return sum(self.compartments_state)

    @property
    def compartments_state(self):
        return np.array([getattr(self, compartment_name) for compartment_name in self.compartments_names])

    def change(self, t, parameters):
        self.t = t
        increments = self.increments_function(self, *parameters)
        if increments is None:
            return
        for compartment_name, increment in zip(self.compartments_names, increments):
            setattr(self, compartment_name, max(getattr(self, compartment_name) + increment, 0))

    def __getattr__(self, item):
        what = getattr(self.data, item)
        if '__iter__' in dir(what):
            return what[what.index(self.t)]
        return what


class Data:
    """  """
    def __init__(self, title: str, time_points: list, **kwargs):
        self.title, self.time_points = title, np.array(time_points)
        for key, value in kwargs.items():
            if '__iter__' in dir(value):
                assert len(time_points) == len(value), 'Different length of time_points and new values'
            setattr(self, key, value)

    def __getattr__(self, item):  # to calm linter
        assert item in self.__dict__, f'{item} is not in {self.__dict__}'
        return getattr(self, item)


class Model:
    """ Main class to describe epidemic model and fit data, can also generate data """
    def __init__(self, compartments_names: list, increments_function: Callable, residuals_function: Callable = mse):
        self.compartments_names = compartments_names
        self.increments_function, self.residuals_function = increments_function, residuals_function
        self.parameters_names = list(inspect.signature(increments_function).parameters)[1:]
        self.k = len(self.parameters_names)

    def generate(self, time_points, compartments_state, parameters, data=None, randomizer=None, randomize_once=True):
        comp_d = {k: v for k, v in zip(self.compartments_names, compartments_state)}
        state = State(time_points[0], self.compartments_names, self.increments_function, data=data, **comp_d)
        results = []
        if randomizer and randomize_once:
            parameters = [randomizer(self.parameters_names[i], parameters[i]) for i in range(self.k)]
        for t in time_points:
            if randomizer and not randomize_once:
                parameters = [randomizer(self.parameters_names[i], parameters[i]) for i in range(self.k)]
            state.change(t, parameters)
            results.append(np.array(state.compartments_state))
        results = np.where(np.isnan(results), 0, results)
        return Result(self, parameters, time_points, np.array(results).T)

    def generate_compartment(self, name, time_points, init_compartments_state, parameters_values, randomizer=None,
                             randomize_once=True, title='GeneratedData'):
        result = self.generate(time_points, init_compartments_state, parameters_values,
                               randomizer=randomizer, randomize_once=randomize_once)
        return Data(title, time_points, **{name: result.compartments_values[self.compartments_names.index(name)]})

    def evaluate(self, results, y_observed, compartment='I'):
        y_predicted = results[self.compartments_names.index(compartment)]
        return self.residuals_function(y_observed, y_predicted)

    def _construct_inner_function_to_minimize(self, data, compartment_state, fit_compartment):
        def inner_function_to_minimize(model_params):
            result = self.generate(data.time_points, compartment_state, model_params, data=data)
            result.set_target(fit_compartment, data.__getattr__(fit_compartment))
            return self.residuals_function(result.observed_values, result.predicted_values)
        return inner_function_to_minimize

    def _construct_outer_function_to_minimize(self, data, compartments_state, param_nums, fit_compartment):
        def outer_function_to_minimize(params):
            params_lists = self._combine_lists(len(data), param_nums, params)
            err = 0
            for i, pars in enumerate(params_lists):
                err += self._construct_inner_function_to_minimize(data[i], compartments_state[i],
                                                                  fit_compartment)(pars)
            return err
        return outer_function_to_minimize

    def _combine_lists(self, n, param_nums, params):
        k_doubling = len(params) - self.k
        copies = param_nums[self.k:self.k + k_doubling]
        params_lists = [[] for _ in range(n)]
        params_lists[0] = list(params[:self.k])
        copies_added = 0
        for i in range(n - 1):
            for j in range(self.k):
                if j in copies:
                    params_lists[1 + i].append(params[self.k * (i + 1) + j - copies_added])
                else:
                    params_lists[1 + i].append(params[j])
                    copies_added += 1
        return params_lists

    def fit(self, init_compartments_state: Union[list[int], list[list[int]]], data, bounds_min: list, bounds_max: list,
            fit_compartment='I', shared_parameters=[], optimizer=lm, params=None):  # from state
        assert init_compartments_state[0], 'compartments_state should be non empty list'
        assert fit_compartment in self.compartments_names, 'fit_compartment should be in compartments_names'
        if not isinstance(data, list):
            data = [data]
        if not isinstance(init_compartments_state[0], list):
            init_compartments_state = [init_compartments_state]
        if len(data) > 1 and len(init_compartments_state) == 1:
            init_compartments_state = list(*init_compartments_state) * len(data)

        bounds = [bounds_min, bounds_max]
        p0 = [(bounds_min[i] + bounds_max[i]) / 2 for i in range(self.k)]
        param_dict = dict(zip(self.parameters_names, p0))
        param_nums = [*range(self.k)]
        N = len(data)
        for i in range(1, N):
            for j, par in enumerate(self.parameters_names):
                if par not in shared_parameters:
                    param_dict[par + str(i + 1)] = p0[j]
                    param_nums.append(j)
                    [b.append(b[j]) for b in bounds]
                    p0.append(p0[j])
        if params:
            params = list(params)
        else:
            function_to_minimize = self._construct_outer_function_to_minimize(data, init_compartments_state,
                                                                              param_nums, fit_compartment)
            params = optimizer(function_to_minimize, p0, bounds)
        params_lists = self._combine_lists(N, param_nums, params)
        all_results = []
        for i, pars in enumerate(params_lists):
            result = self.generate(data[i].time_points, init_compartments_state[i], pars, data=data[i])
            result.set_target(fit_compartment, data[i].__getattr__(fit_compartment), data[i].title)
            all_results.append(result)
        return ResultsList(all_results) if len(all_results) > 1 else all_results[0]
