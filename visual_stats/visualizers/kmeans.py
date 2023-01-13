import p5
import numpy as np

from functools import reduce
from scipy.spatial.distance import euclidean
from typing import Tuple

from .visualizer import Visualizer, VisualizerState
from ..models import KMeans


class KMeansVisualizer(Visualizer):
    RADIUS = 10
    COLOURS = [
        {'r': 255, 'g': 0, 'b': 0},
        {'r': 0, 'g': 255, 'b': 0},
        {'r': 0, 'g': 0, 'b': 255}
    ]

    def __init__(self, points: np.ndarray, model: KMeans):
        super().__init__()

        self.points = points

        self.min_x = np.amin(points[:,0])
        self.max_x = np.amax(points[:,0])
        self.x_padding = 0.1 * self.WIDTH

        self.min_y = np.amin(self.points[:,1])
        self.max_y = np.amax(self.points[:,1])
        self.y_padding = 0.1 * self.HEIGHT

        self.model = model

        self.states = {
            VisualizerState.INITIAL: [
                self._draw_points,
                self._draw_centroids,
                self._draw_boundaries,
                self._colour_classes
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
        self.model.train(self.points)

    def _draw_points(self):
        self._background()

        points = self.points

        for point in points:
            coordinate = self._to_screen_coordinate(point)
            p5.circle(coordinate, self.RADIUS)

        return points

    def _draw_centroids(self, draw_points: bool = True):
        points = self._draw_points() if draw_points else self.points

        centroids = self.model.centroids

        for i in range(len(centroids)):
            colour = self.COLOURS[i]
            p5.stroke(r=colour['r'], g=colour['g'], b=colour['b'])
            p5.fill(r=colour['r'], g=colour['g'], b=colour['b'])
            coordinates = self._to_screen_coordinate(centroids[i])
            p5.circle(coordinates, self.RADIUS)

        return points, centroids

    def _draw_boundaries(self):
        points, centroids = self._draw_centroids()

        midpoint = lambda *args: np.mean(np.stack([*args]), axis=0)

        midpoints = [[midpoint(centroids[i], centroids[j]) \
                      for j in range(i, len(centroids)) if i != j] \
                     for i in range(len(centroids))]
        midpoints = reduce(lambda x, y: x + y, midpoints)

        def circumcenter(a: np.ndarray, b: np.ndarray, c: np.ndarray):
            d = 2 * (a[0] * (b[1] - c[1]) + \
                     b[0] * (c[1] - a[1]) + \
                     c[0] * (a[1] - b[1]))

            x = ((a[0] ** 2 + a[1] ** 2) * (b[1] - c[1]) + \
                 (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1]) + \
                 (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])) / d
            y = ((a[0] ** 2 + a[1] ** 2) * (c[0] - b[0]) + \
                 (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0]) + \
                 (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])) / d
            return (x, y)

        print(centroids)
        center = circumcenter(*centroids)
        print(center)

        corners = {
            'leftbot': (np.amin(self.points[:,0]), np.amin(self.points[:,1])),
            'lefttop': (np.amin(self.points[:,0]), np.amax(self.points[:,1])),
            'rightbot': (np.amax(self.points[:,0]), np.amin(self.points[:,1])),
            'righttop': (np.amax(self.points[:,0]), np.amax(self.points[:,1]))
        }
        edges = {
            'top': np.asarray((corners['lefttop'], corners['righttop'])),
            'bot': np.asarray((corners['leftbot'], corners['rightbot'])),
            'right': np.asarray((corners['righttop'], corners['rightbot'])),
            'left': np.asarray((corners['lefttop'], corners['leftbot']))
        }
        directions = [
            {-1: 'left', 1: 'right'},
            {-1: 'bot', 1: 'top'}
        ]

        def intersect(v1: Tuple[np.ndarray], v2: Tuple[np.ndarray]):
            da = v1[1] - v1[0]
            db = v2[1] - v2[0]
            dp = v1[0] - v2[0]
            dap = np.asarray([-da[1], da[0]])
            denom = np.dot( dap, db)
            num = np.dot( dap, dp )
            return (num / denom.astype(float))*db + v2[0]

        p5.stroke(r=128, g=128, b=255)
        p5.stroke_weight(3)

        for midpoint in midpoints:
            p5.circle(self._to_screen_coordinate(midpoint), self.RADIUS)
            vector = midpoint - center
            direction = np.sign(vector).astype(int)
            direction = [directions[dim][direction[dim]] \
                         for dim in range(len(direction))]
            intersections = [intersect(edges[direction[i]],
                                       (midpoint, center)) \
                             for i in range(len(directions))]
            distances = [euclidean(center, intersection) \
                         for intersection in intersections]
            intersection = intersections[np.argmin(distances)]

            p5.line(self._to_screen_coordinate(center),
                    self._to_screen_coordinate(intersection))

        return points, centroids

    def _colour_classes(self):
        points, centroids = self._draw_boundaries()

        for point in points:
            coordinate = self._to_screen_coordinate(point)
            cluster = self.model.predict(point)
            colour = self.COLOURS[cluster]
            p5.stroke(r=colour['r'] // 2, g=colour['g'] // 2,
                      b=colour['b'] // 2)
            p5.fill(r=colour['r'] // 2, g=colour['g'] // 2, b=colour['b'] // 2)
            p5.circle(coordinate, self.RADIUS)

        self._draw_centroids(draw_points=False)
