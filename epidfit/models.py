from epidfit import State


'''Simplest one compartment models'''


def linear(state: "State", change: float):
    return [change]


def quadratic(state: "State", a: float, b: float):
    return [a * state.compartments_state[0] + b]


def exponential(state: "State", a: float):
    return [a * state.compartments_state[0]]


'''Simple compartment models'''


def si(state: "State", beta: float):
    change = beta * state.S
    return [-change, change]


def sis(state: "State", beta: float, gamma: float):
    infected = beta * state.S * state.I / state.N
    recovered = gamma * state.I
    return [-infected + recovered, infected - recovered]


def sir(state: "State", beta: float, gamma: float):
    infected = beta * state.S * state.I / state.N
    recovered = gamma * state.I
    return [-infected, infected - recovered, recovered]


def seir(state: "State", beta: float, sigma: float, gamma: float):
    exposed = beta * state.S * state.I / state.N
    infected = sigma * state.E
    recovered = gamma * state.I
    return [-exposed, exposed - infected, infected - recovered, recovered]
