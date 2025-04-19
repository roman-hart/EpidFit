from epidfit import Data, Model
from epidfit.models import linear  # simple predefined model

'''All needed variables'''
compartments_names = ['I']
time_points = [1, 2, 3, 4, 5]
data_points = {'I': [1, 2, 3, 3, 5]}  # infected
initial_state = [data_points['I'][0]]  # initial susceptible and infected
bounds_min, bounds_max = [0], [10]

'''Creation of necessary objects: Model and Data'''
model = Model(compartments_names=compartments_names, increments_function=linear)
data = Data(time_points=time_points, **data_points)

'''Fitting process and evaluation'''
result = model.fit(init_compartments_state=initial_state, data=data, bounds_min=bounds_min, bounds_max=bounds_max)
print('Optimal parameters:', result.parameters)
result.plot()
