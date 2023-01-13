import argparse
import p5
import numpy as np
import pandas as pd

from visual_stats import KeyHandler, LinearRegression, \
                         LinearRegressionVisualizer


def _generate_data(n: int, inputs: str, outputs: str) -> pd.DataFrame:
    x = np.random.uniform(0, 1, n)
    y = x + np.random.normal(0, 0.1, n)

    return pd.DataFrame({inputs: x, outputs: y})

def visualize_linear_regression(data: str, inputs: str, outputs: str, n: int,
                                learning_rate: float, fps: float,
                                key_handler: KeyHandler):
    if data is None:
        data = _generate_data(n, inputs=inputs, outputs=outputs)

    x = data[inputs]
    y = data[outputs]

    model = LinearRegression(np.mean(y), 0, learning_rate=learning_rate)
    visualizer = LinearRegressionVisualizer(x, y, model=model)
    key_handler.register(visualizer)

    p5.run(sketch_draw=visualizer.draw, frame_rate=fps)


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
    parser.add_argument('-lr', '--learning_rate', required=False, type=float,
                        default=0.5, help='Learning rate used by the model')
    parser.add_argument('-fps', '--frames_per_second', required=False,
                        type=float, default=60, help='FPS used for rendering')

    args = parser.parse_args()

    key_handler = KeyHandler()

    def key_pressed(event):
        key_handler(event.key)

    visualize_linear_regression(data=args.data,
                                inputs=args.inputs,
                                outputs=args.outputs,
                                n=args.size,
                                learning_rate=args.learning_rate,
                                fps=args.frames_per_second,
                                key_handler=key_handler)
