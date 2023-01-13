import p5
import numpy as np

from typing import Tuple

from .visualizer import Visualizer, VisualizerState
from ..models import LinearRegression


class LinearRegressionVisualizer(Visualizer):
    RADIUS = 10

    def __init__(self, x: np.ndarray, y: np.ndarray, model: LinearRegression):
        super().__init__()

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

        self.model = model

        self.states = {
            VisualizerState.INITIAL: [
                self._draw_points,
                self._draw_model,
                self._draw_losses
            ],
            VisualizerState.LOOP: [
                self._loop
            ]
        }

    def _to_screen_coordinate(self, point: Tuple[int], pad_x: bool = True, pad_y: bool = True):
        x_padding = self.x_padding if pad_x else 0
        y_padding = self.y_padding if pad_y else 0

        x = (point[0] - self.min_x) / (self.max_x - self.min_x) * \
            (self.WIDTH - 2 * x_padding)
        y = (point[1] - self.min_y) / (self.max_y - self.min_y) * \
            (self.HEIGHT - 2 * y_padding)

        return (x_padding + x, self.HEIGHT - (y_padding + y))

    def _update(self):
        self.model.train(self.x, self.y)

    def _draw_points(self):
        self._background()

        p5.stroke(255)
        p5.stroke_weight(1)
        p5.fill(255)

        points = [(self.x[i], self.y[i]) for i in range(len(self.x))]
        coordinates = [self._to_screen_coordinate(point) for point in points]

        for coordinate in coordinates:
            p5.circle(coordinate, self.RADIUS)

        return points, coordinates

    def _draw_model(self):
        points, coordinates = self._draw_points()

        start = self.model.predict(np.amin(self.x))
        start = self._to_screen_coordinate((np.amin(self.x), start), pad_x=False, pad_y=False)
        end = self.model.predict(np.amax(self.x))
        end = self._to_screen_coordinate((np.amax(self.x), end), pad_x=False, pad_y=False)

        p5.stroke(r=128, g=128, b=255)
        p5.stroke_weight(3)
        p5.line(start, end)

        p5.stroke(255)
        p5.fill(255)
        p5.stroke_weight(1)
        p5.text_font(self.DEFAULT_FONT, size=15)

        text_x = (np.amax(self.x) - np.amin(self.x)) * 0.3
        text_y = np.amin(self.y)
        text_coordinate = self._to_screen_coordinate((text_x, text_y))
        p5.text((f'y={round(self.model.intercept, 2):.2f} + '
                 f'{round(self.model.coefficient, 2):.2f}x'),
                 text_coordinate[0], text_coordinate[1])

        return points, coordinates

    def _draw_losses(self):
        _, coordinates = self._draw_model()

        p5.stroke(255)
        p5.stroke_weight(1)

        total_loss = 0

        for i in range(len(coordinates)):
            prediction = self.model.predict(self.x[i])
            ground_truth = self._to_screen_coordinate((self.x[i], prediction))
            p5.stroke(255)
            p5.line(ground_truth, coordinates[i])
            loss = (self.y[i] - prediction) ** 2
            total_loss += loss
            loss = 1 - (loss / self.max_loss)
            loss = loss ** 3
            r, g, b = (255, int(255 * loss), int(255 * loss))
            p5.stroke(r=r, g=g, b=b)
            p5.fill(r=r, g=g, b=b)
            p5.circle(coordinates[i], self.RADIUS)

        p5.stroke(255)
        p5.fill(255)
        p5.stroke_weight(1)
        p5.text_font(self.DEFAULT_FONT, size=15)

        text_x = (np.amax(self.x) - np.amin(self.x)) * 0.6
        text_y = np.amin(self.y)
        text_coordinate = self._to_screen_coordinate((text_x, text_y))
        p5.text(f'loss={round(total_loss, 2):.2f}', text_coordinate[0],
                text_coordinate[1])



