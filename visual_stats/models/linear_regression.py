import numpy as np

from typing import Union

class LinearRegression():
    def __init__(self, intercept: float, coefficient: float,
                 learning_rate: float):
        self.intercept = intercept
        self.coefficient = coefficient
        self.learning_rate = learning_rate

    def predict(self, x: Union[float, np.ndarray]) -> float:
        return self.intercept + x * self.coefficient

    def train(self, x: np.ndarray, y: np.ndarray):
        predictions = self.predict(x)
        d_i = ((-2 / len(x)) * np.sum(y - predictions))
        2 * (y - predictions) * -x
        d_c = ((1 / len(x)) * np.sum(2 * (y - predictions) * -x))
        self.intercept = self.intercept - self.learning_rate * d_i
        self.coefficient = self.coefficient - self.learning_rate * d_c
