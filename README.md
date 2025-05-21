# EpidFit

**EpidFit** is a Python library for modeling infectious disease dynamics using compartmental models. It provides tools to define models, fit them to real-world data, and analyze results with customizable functions for optimization, residuals, and randomization.

---

## ✨ Quick Start

### 1. Define Your Model Function

You need to define an **increment function** describing how compartments (e.g., Susceptible, Infected) change over time. For example, a simplified SIS model:

```python
def sis(state: "State", beta: float, gamma: float):
    infected = beta * state.S * state.I / state.N
    recovered = gamma * state.I
    return [-infected + recovered, infected - recovered]
```

Alternatively, you can directly modify the state:

```python
def sis(state: "State", beta: float, gamma: float):
    infected = beta * state.S * state.I / state.N
    recovered = gamma * state.I
    state.S = state.S - infected + recovered
    state.I = state.I + infected - recovered
```

---

### 2. Define a Residual Function (optional)

Used to evaluate model fit. Default: Ordinary Least Squares.

```python
def residuals(y_observed: np.array, y_predicted: np.array):
    return np.sum((y_observed - y_predicted) ** 2)
```

---

### 3. Define an Optimizer (optional)

Custom function to optimize parameters.

```python
def optimizer(function_to_minimize, init_params, bounds):
    return minimize(function_to_minimize,
                    x0=init_params,
                    method='Powell',
                    bounds=zip(*bounds)).x
```

---

### 4. Define a Randomizer (optional)

Use this to inject random variation into parameters when simulating.

```python
def randomizer(parameter_name: str, value: float):
    if parameter_name == 'beta':
        return np.random.normal(value, value / 10)
    return value
```

---

### 5. Create the Model

```python
model = Model(compartments_names=['S', 'I'], increments_function=sis)
```

---

### 6. Load or Generate Data

If you already have data:

```python
data = Data(title='MyData', time_points=xs, I=ys)
```

You can also attach extra information:

```python
data.vaccination_start = 1960
```

Or generate data from the model:

```python
generated_data = model.generate_compartment(
    name='I',
    time_points=range(20),
    init_compartments_state=[99, 1],
    parameters_values=[2.01, 2],
    randomizer=randomizer
)
```

---

### 7. Fit the Model to Data

```python
result = model.fit(
    init_compartments_state=[99, 1],
    data=data,
    bounds_min=[1, 1],
    bounds_max=[3, 3],
    optimizer=optimizer
)
```

For multiple datasets with shared parameters:

```python
result = model.fit(
    init_compartments_state=[[99, 1], [99, 1]],
    data=[data1, data2],
    bounds_min=[1, 1],
    bounds_max=[3, 3],
    shared_parameters=['T'],
    optimizer=optimizer
)
```

---

### 8. Analyze Results

```python
print('Optimal parameters:', result.parameters)
print('Observed values:', result.observed_values)
print('Predicted values:', result.predicted_values)

result.plot(compartments_names=['I'],
            title='Plot title',
            center_text=f'R2: {result.r2:.2f}')

result.stability_histogram(
    sigma=0.1, func='residuals',
    n_sim=1000, n_bins=50)

result.heatmap(
    parameter_x='beta', parameter_y='gamma',
    sigma_x=0.1, sigma_y=0.1,
    func='r2', n_sim=1000)
```

---


See more in [examples](https://github.com/roman-hart/EpidFit/tree/master/examples).

---


## 📄 License

MIT License.
