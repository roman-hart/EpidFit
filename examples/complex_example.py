from epidfit.main import Data, Model
import numpy as np
from scipy.optimize import minimize  # to configure custom optimizer


'''Useful variables and functions'''
compartments_names = ['S', 'I']
time_points = [1, 2, 3, 4, 5]
data_points = {'I': [1, 2, 3, 3, 5]}
N = 100  # population
initial_infected = data_points['I'][0]
initial_state = [N - initial_infected, initial_infected]  # initial susceptible and infected
bounds_min, bounds_max = [0], [10]


def simplified_sis(state, beta: float, gamma: float):  # Model() increments_function
    infected = beta * state.S
    recovered = gamma * state.I
    return [-infected + recovered, infected - recovered]


def residuals(y_observed: np.array, y_predicted: np.array):  # Model() residuals_function - optional
    return np.sum((y_observed - y_predicted) ** 2)


def randomizer(parameter_name: str, value: float):  # model.generate_compartment() - optional
    if parameter_name == 'beta':
        return np.random.normal(value, value / 10)
    return value


def optimizer(function_to_minimize, init_params, bounds: list[list, list]):  # model.fit() - optional
    return minimize(function_to_minimize, x0=init_params, method='Powell', bounds=zip(*bounds)).x


'''Creation of all needed objects: Model and Data instances'''
model = Model(compartments_names=compartments_names, increments_function=simplified_sis)
data = Data(title="My Data", time_points=time_points, **data_points)
generated_data = model.generate_compartment(name='I',
                                            time_points=range(20),
                                            init_compartments_state=[99, 1],
                                            parameters_values=[2.01, 2, ],
                                            randomizer=randomizer)

'''Fitting process and evaluation of results'''
result = model.fit(init_compartments_state=[[99, 1], [99, 1]],
                   data=[data, generated_data],
                   bounds_min=[1, 1],
                   bounds_max=[3, 3],
                   shared_parameters=['T'],
                   optimizer=optimizer)

print('Optimal parameters:', result.parameters)
print('Observed values:', result.observed_values)
print('Predicted values:', result.predicted_values)
result.plot(title='My Data 2')  # todo title
