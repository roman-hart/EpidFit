import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class Result:
    def __init__(self, model: "Model", parameters_values: list[float], time_points: list[float],
                 compartments_values: list[list]):
        self.model, self.parameters, self.time_points, self.generated_compartments = model, parameters_values, time_points, compartments_values
        self.n, self.k = len(time_points), len(parameters_values)
        self.target_compartment_name, self.observed_values, self.predicted_values, self.subscription = None, None, None, ''

    def set_target(self, name: str, values: list, subscription=''):
        self.target_compartment_name, self.observed_values, self.subscription = name, values, subscription
        self.predicted_values = self.generated_compartments[self.model.compartments_names.index(self.target_compartment_name)]

    @property
    def residuals(self):
        return self.model.residuals_function(self.observed_values, self.predicted_values)

    @property
    def r2(self):
        rss = np.sum((self.observed_values - self.predicted_values) ** 2)
        tss = np.sum((self.observed_values - np.mean(self.predicted_values)) ** 2)
        return 1 - (rss / tss)

    @property
    def r2adj(self):
        k, n = len(self.model.parameters_names), len(self.observed_values)
        return 1 - ((1 - self.r2) * (n - 1)) / (n - k - 1)

    @property
    def mse(self):
        return np.mean((self.observed_values - self.predicted_values) ** 2)

    @property
    def rmse(self):
        return np.sqrt(self.mse())

    @property
    def mae(self):
        return np.mean(np.abs(self.observed_values - self.predicted_values))

    @property
    def mape(self):
        return np.mean(np.abs((self.observed_values - self.predicted_values) / self.observed_values)) * 100

    @property
    def bic(self):
        n = len(self.observed_values)
        mse = self.mse()
        return n * np.log(mse) + self.k * np.log(n)

    @property
    def aic(self):
        n = len(self.observed_values)
        mse = self.mse()
        return 2 * self.k + n * np.log(mse)

    @property
    def init_state(self):
        return self.generated_compartments.T[0]

    def plot(self, compartments_names: list[str] = None, center_text: str = None, title=None, xtitle='years',
             ytitle='cases per 100000 people'):
        targeted = self.observed_values is not None
        if compartments_names is None:
            compartments_names = [self.target_compartment_name] if targeted else [self.model.compartments_names]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        x = self.time_points
        base_y, base_color = None, 'blue'
        for i, name in enumerate(compartments_names):
            y = self.generated_compartments[self.model.compartments_names.index(name)]
            line_dict = dict(color=base_color, width=2) if name == self.target_compartment_name else dict(width=2)
            text = f'compartment {name}'
            if i == 0:
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=text, line=line_dict))
                fig.update_yaxes(title_text=text, secondary_y=False)
                base_y = y
            else:
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=text, line=line_dict), secondary_y=True)
                fig.update_yaxes(title_text=text, secondary_y=True)
        if targeted:
            fig.add_trace(go.Scatter(x=x, y=self.observed_values, mode='markers',
                                     marker=dict(color=base_color, size=7), name=f'observed {self.target_compartment_name}'))
        if center_text:
            fig.add_annotation(x=(min(x) + max(x)) / 2, y=(min(base_y) + max(base_y)) / 2, text=center_text,
                               font=dict(family="Arial, sans-serif", size=16, color='red'), showarrow=False)
        if title is None:
            title = self.subscription
        fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle, legend_title_text='', height=300)
        fig.update_layout(margin=dict(l=1, r=1, t=33, b=1))
        fig.show()

    def stability_histogram(self, sigma=0.1, func='residuals', n_sim=1000, n_bins=100):
        assert hasattr(self, func), f"'{type(self).__name__}' object has no attribute '{func}'"
        results = []
        original_predicted_values = self.predicted_values.copy()
        func_values = eval(f"self.{func}")
        for _ in range(n_sim):
            sampled_parameters = [p * (1 + np.random.normal(0, sigma)) for p in self.parameters]
            self.predicted_values = self.model.generate_compartment(
                self.target_compartment_name, self.time_points, self.init_state, sampled_parameters
            ).__getattr__(self.target_compartment_name)
            results.append(eval(f"self.{func}"))  # Use the dynamically set function
        self.predicted_values = original_predicted_values  # Restore original predicted values
        title = f"Stability Analysis of {func} for {self.subscription}" if self.subscription else f"Stability Analysis of {func}"
        fig = px.histogram(np.array(results), nbins=n_bins, title=title, histnorm='probability')
        fig.update_layout(xaxis_title=f"Values of {func}", yaxis_title="Relative Frequency")
        # Add the vertical line
        fig.add_shape(
            type="line",
            x0=float(func_values),
            x1=float(func_values),
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", dash="dash", width=2)
        )
        # Add a "dummy" trace for the legend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name="Current Value"
            )
        )
        fig.update_traces(name="Distribution", selector=dict(type='histogram'))
        fig.update_layout(margin=dict(l=1, r=1, t=33, b=1), height=300, legend_title_text="", showlegend=True)
        fig.show()

    def heatmap(self, parameter_x: str, parameter_y: str, sigma_x=0.1, sigma_y=0.1, func='residuals', n_sim=1000):
        """
        Generates a heatmap by systematically varying two parameters (default function: residuals).
        """
        assert hasattr(self, func), f"'{type(self).__name__}' object has no attribute '{func}'"
        parameters_names = self.model.parameters_names
        assert parameter_x in parameters_names, f'The model has no {parameter_x} parameter'
        assert parameter_y in parameters_names, f'The model has no {parameter_y} parameter'
        n_sim = int(n_sim ** 0.5)
        parameter_x_index, parameter_y_index = parameters_names.index(parameter_x), parameters_names.index(parameter_y)
        original_predicted_values = self.predicted_values.copy()
        original_parameters = self.parameters.copy()
        x_values = np.linspace(original_parameters[parameter_x_index] * (1 - sigma_x),
                               original_parameters[parameter_x_index] * (1 + sigma_x),
                               n_sim)
        y_values = np.linspace(original_parameters[parameter_y_index] * (1 - sigma_y),
                               original_parameters[parameter_y_index] * (1 + sigma_y),
                               n_sim)

        func_values = np.zeros((n_sim, n_sim))
        for i, x_val in enumerate(x_values):
            for j, y_val in enumerate(y_values):
                params = original_parameters.copy()
                params[parameter_x_index] = x_val
                params[parameter_y_index] = y_val
                self.predicted_values = self.model.generate_compartment(
                    self.target_compartment_name, self.time_points, self.init_state, params
                ).__getattr__(self.target_compartment_name)
                func_value = eval(f"self.{func}")
                func_values[i, j] = np.mean(func_value)  # Store mean value
        self.predicted_values = original_predicted_values
        data_dict = {"X": np.repeat(x_values, len(y_values)),  # Expand x-values
                     "Y": np.tile(y_values, len(x_values)),  # Expand y-values
                     "Func_Value": func_values.flatten()}  # Flatten 2D array to 1D
        title = f"Heatmap of {func} over {parameter_x} & {parameter_y}"
        if self.subscription:
            title += f" for {self.subscription}"
        fig = px.density_heatmap(
            data_dict, x="X", y="Y", z="Func_Value",
            histfunc="avg",  # Compute mean instead of sum
            title=title,
            labels={"X": parameter_x, "Y": parameter_y, "Func_Value": func}
        )
        fig.update_layout(margin=dict(l=1, r=1, t=33, b=1))
        fig.show()


class ResultsList(list):
    def __init__(self, list_of_results: list["Result"]):
        super().__init__(list_of_results)

    @property
    def weights(self):
        return np.array([r.n for r in self])

    def __getattr__(self, name):
        """Dynamically retrieve attributes and methods from Result objects."""
        if not self:
            raise AttributeError(f"ResultsList is empty, cannot access '{name}'")

        attr = getattr(self[0], name, None)
        if attr is None:
            raise AttributeError(f"'{type(self[0]).__name__}' has no attribute '{name}'")

        if name in ['model', 'parameters', 'time_points', 'n', 'observed_values', 'predicted_values']:
            return [getattr(r, name) for r in self]

        if not callable(attr):  # If it's a property or attribute → Compute weighted average
            return np.average([getattr(r, name) for r in self], weights=self.weights)

        def method(*args, **kwargs):  # If it's a method → Execute for each Result
            [getattr(r, name)(*args, **kwargs) for r in self]

        return method