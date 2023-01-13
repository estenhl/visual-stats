import argparse
import p5
import numpy as np
import pandas as pd

from visual_stats import KeyHandler, KMeans, KMeansVisualizer


def _generate_data(n: int, x: str, y: str, clusters: int) -> pd.DataFrame:
    memberships = np.random.randint(0, clusters, n)
    centers = np.random.uniform(0, 1, (clusters, 2))

    df = pd.DataFrame({x: 0, y: 0, 'cluster': memberships})

    for i in range(clusters):
        spread = np.random.uniform(0.05, 0.1)
        count = len(np.where(memberships == i)[0])
        x_noise = np.random.normal(0, spread, count)
        y_noise = np.random.normal(0, spread, count)

        df.loc[df['cluster'] == i, x] = centers[i, 0] + x_noise
        df.loc[df['cluster'] == i, y] = centers[i, 1] + y_noise

    return df

def visualize_kmeans(data: str, x: str, y: str, n: int,
                     clusters: int, fps: float, key_handler: KeyHandler):
    if clusters > 3:
        raise NotImplementedError('Not implemented for >3 clusters')


    if data is None:
        data = _generate_data(n, x=x, y=y, clusters=clusters)

    points = np.asarray(list(zip(data[x], data[y])))

    model = KMeans(points, clusters=clusters)
    visualizer = KMeansVisualizer(points, model=model)

    key_handler.register(visualizer)

    p5.run(sketch_draw=visualizer.draw, frame_rate=fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Visualizes fitting kmeans clustering'))

    parser.add_argument('-d', '--data', required=False, default=None,
                        help='Optional data to use. Should be a CSV-file')
    parser.add_argument('-x', '--x', required=False, default='x',
                        help=('Optional x-coordinate. Should be used if '
                              'data is given, and refer to a named column'))
    parser.add_argument('-y', '--y', required=False, default='y',
                        help=('Optional y-coordinate. Should be used if '
                              'data is given, and refer to a named column'))
    parser.add_argument('-n', '--size', required=False, type=int, default=100,
                        help='Number of datapoints to use')
    parser.add_argument('-c', '--clusters', required=True, type=int,
                        help='Number of clusters to use')
    parser.add_argument('-fps', '--frames_per_second', required=False,
                        type=float, default=60, help='FPS used for rendering')

    args = parser.parse_args()

    key_handler = KeyHandler()

    def key_pressed(event):
        key_handler(event.key)

    visualize_kmeans(data=args.data,
                     x=args.x,
                     y=args.y,
                     n=args.size,
                     clusters=args.clusters,
                     fps=args.frames_per_second,
                     key_handler=key_handler)
