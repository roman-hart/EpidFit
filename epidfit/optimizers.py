from scipy.optimize import minimize
import pyswarms as ps


def _scipy(function_to_minimize, init_params, bounds, method='BFGS', options={'maxiter': 10000, 'disp': False}):
    return minimize(function_to_minimize, init_params, method=method, bounds=zip(*bounds), options=options).x


def lm(function_to_minimize, init_params, bounds):
    return _scipy(function_to_minimize, init_params, bounds, method='L-BFGS-B')


def slsqp(function_to_minimize, init_params, bounds):
    return _scipy(function_to_minimize, init_params, bounds, method='SLSQP')


def tnc(function_to_minimize, init_params, bounds):
    return _scipy(function_to_minimize, init_params, bounds, method='TNC')


def powell(function_to_minimize, init_params, bounds):
    return _scipy(function_to_minimize, init_params, bounds, method='Powell')


def nm(function_to_minimize, init_params, bounds):
    return _scipy(function_to_minimize, init_params, bounds, method='Nelder-Mead')


def pso(function_to_minimize, init_params, bounds):

    def function_to_minimize2(params):
        return [function_to_minimize(p) for p in params]

    optimizer = ps.single.GlobalBestPSO(n_particles=40,
                                        dimensions=len(init_params),
                                        options={'c1': 0.8, 'c2': 0.5, 'w': 1.0},
                                        bounds=bounds)
    cost, params = optimizer.optimize(function_to_minimize2, iters=1000)
    return params


def generate_pso(n_particles=40, c1=0.8, c2=0.5, w=1.0, iters=1000):
    def pso(function_to_minimize, init_params, bounds):
        def function_to_minimize2(params):
            return [function_to_minimize(p) for p in params]

        optimizer = ps.single.GlobalBestPSO(n_particles=n_particles,
                                            dimensions=len(init_params),
                                            options={'c1': c1, 'c2': c2, 'w': w},
                                            bounds=bounds)
        cost, params = optimizer.optimize(function_to_minimize2, iters=iters)
        return params

    return pso
