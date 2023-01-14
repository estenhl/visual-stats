import p5
import numpy as np

from functools import reduce
from scipy.spatial.distance import euclidean
from typing import Tuple

from .visualizer import Visualizer, VisualizerState
from ..models import KMeans
from ..utils.linalg import circumcenter, intersect


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

        midpoints = [midpoint(centroids[i], centroids[j]) \
                     for i, j in [(0, 1), (0, 2), (1, 2)]]
        distances = [euclidean(centroids[i], centroids[j]) \
                     for i, j in [(0, 1), (0, 2), (1, 2)]]

        center = circumcenter(*centroids)

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
        opposite = {
            'top': 'bot',
            'bot': 'top',
            'right': 'left',
            'left': 'right'
        }

        def inside(triangle: np.ndarray, point: np.ndarray) -> bool:
            (x1, y1), (x2, y2), (x3, y3) = triangle

            c1 = (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)
            c2 = (x3 - x2) * (point[1] - y2) - (y3 - y2) * (point[0] - x2)
            c3 = (x1 - x3) * (point[1] - y3) - (y1 - y3) * (point[0] - x3)

            return (c1<0 and c2<0 and c3<0) or (c1>0 and c2>0 and c3>0)


        p5.stroke(r=128, g=128, b=255)
        p5.stroke_weight(3)

        for midpoint in midpoints:
            vector = midpoint - center
            direction = np.sign(vector).astype(int)
            direction = [directions[dim][direction[dim]] \
                         for dim in range(len(direction))]

            others = [m for m in midpoints if not np.array_equal(m, midpoint)]

            if inside(others + [center], midpoint):
                direction = [opposite[dim] for dim in direction]

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
