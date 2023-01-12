import argparse
import p5
import numpy as np
import pandas as pd

from PIL import ImageFont
from typing import Tuple, Union

class LinearRegression():
    def __init__(self, intercept: float, coefficient: float,
                 learning_rate: float = 0.1):
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


class LinearRegressionVisualizer():
    WIDTH = 640
    HEIGHT = 320
    RADIUS = 10

    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.min_x = np.amin(x)
        self.max_x = np.amax(x)
        self.x_padding = 0.1 * self.WIDTH

        self.min_y = np.amin(y)
        self.max_y = np.amax(y)
        self.y_padding = 0.1 * self.HEIGHT

        self.max_loss = max(abs(np.amax(y) - np.mean(y)),
                            abs(np.amin(y) - np.mean(y))) ** 2

        self.model = LinearRegression(np.mean(y), 0)

    def _to_screen_coordinate(self, point: Tuple[int], pad_x: bool = True, pad_y: bool = True):
        x_padding = self.x_padding if pad_x else 0
        y_padding = self.y_padding if pad_y else 0

        x = (point[0] - self.min_x) / (self.max_x - self.min_x) * \
            (self.WIDTH - 2 * x_padding)
        y = (point[1] - self.min_y) / (self.max_y - self.min_y) * \
            (self.HEIGHT - 2 * y_padding)

        return (x_padding + x, self.HEIGHT - (y_padding + y))

    def draw(self):
        p5.size(640, 320)
        p5.background(0)
        p5.stroke(255)

        points = [(self.x[i], self.y[i]) for i in range(len(self.x))]
        coordinates = [self._to_screen_coordinate(point) for point in points]

        for coordinate in coordinates:
            p5.circle(coordinate, self.RADIUS)

        points = [(self.x[i], self.y[i]) for i in range(len(self.x))]
        coordinates = [self._to_screen_coordinate(point) for point in points]

        start = self.model.predict(np.amin(self.x))
        start = self._to_screen_coordinate((np.amin(self.x), start), pad_x=False, pad_y=False)
        end = self.model.predict(np.amax(self.x))
        end = self._to_screen_coordinate((np.amax(self.x), end), pad_x=False, pad_y=False)

        p5.stroke(r=128, g=128, b=255)
        p5.stroke_weight(3)
        p5.line(start, end)
        p5.stroke(255)
        p5.stroke_weight(1)

        for i in range(len(coordinates)):
            prediction = self.model.predict(self.x[i])
            ground_truth = self._to_screen_coordinate((self.x[i], prediction))
            p5.stroke(255)
            p5.line(ground_truth, coordinates[i])
            loss = (self.y[i] - prediction) ** 2
            loss = 1 - (loss / self.max_loss)
            loss = loss ** 3
            r, g, b = (255, int(255 * loss), int(255 * loss))
            p5.stroke(r=r, g=g, b=b)
            p5.fill(r=r, g=g, b=b)
            p5.circle(coordinates[i], self.RADIUS)

        p5.stroke(255)
        p5.fill(255)
        #p5.text_font(p5.load_font('arial'), size=10)
        text_coordinate = self._to_screen_coordinate((0.75, 0.25))
        p5.text(f'y={round(self.model.intercept, 2)} + {round(self.model.coefficient, 2)}x', text_coordinate[0], text_coordinate[1])

        self.model.train(self.x, self.y)


def _generate_data(n: int, inputs: str, outputs: str):
    x = np.random.uniform(0, 1, n)
    y = x + np.random.normal(0, 0.1, n)

    return pd.DataFrame({inputs: x, outputs: y})


def visualize_linear_regression(data: str, inputs: str, outputs: str,
                                n: int, epochs: int):
    if data is None:
        data = _generate_data(n, inputs=inputs, outputs=outputs)

    visualizer = LinearRegressionVisualizer(data[inputs].values,
                                            data[outputs].values)

    p5.run(sketch_draw=visualizer.draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Visualizes fitting a linear regression '
                                      'model with one regressor using '
                                      'gradient descent'))

    parser.add_argument('-d', '--data', required=False, default=None,
                        help='Optional data to use. Should be a CSV-file')
    parser.add_argument('-x', '--inputs', required=False, default='x',
                        help=('Optional input variable. Should be used if '
                              'data is given, and refer to a named column'))
    parser.add_argument('-y', '--outputs', required=False, default='y',
                        help=('Optional output variable. Should be used if '
                              'data is given, and refer to a named column'))
    parser.add_argument('-n', '--size', required=False, type=int, default=50,
                        help='Number of datapoints to use')
    parser.add_argument('-e', '--epochs', required=True, type=int,
                        help='Number of epochs to optimize for')

    args = parser.parse_args()

    visualize_linear_regression(data=args.data,
                                inputs=args.inputs,
                                outputs=args.outputs,
                                n=args.size,
                                epochs=args.epochs)


