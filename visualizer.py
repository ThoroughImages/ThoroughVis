import os
import math
import uuid
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from model_loader import MetaGraphLoader

class Visualizer(object):
    def __init__(self, meta_graph, output_dir=None):
        """Creates a `Visualizer` object.
        Args:
            meta_graph: meta graph
            output_dir: Directory to save the figure
        """
        self._meta_graph = meta_graph
        self._loader = MetaGraphLoader(restore_from=meta_graph)
        self._loader.load()
        if output_dir is None:
            self._output_dir = os.path.join('/tmp', str(uuid.uuid4()))
        else:
            self._output_dir = output_dir
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

    def _prime_powers(self, n):
        """Compute the factors of a positive integer.

        Args:
            n: An integer.

        Returns:
            A set containing all the factors.
        """
        factors = set()
        for x in xrange(1, int(math.sqrt(n)) + 1):
            if n % x == 0:
                factors.add(int(x))
                factors.add(int(n // x))
        return sorted(factors)

    def _get_grid_dim(self, x):
        """Transforms x into product of two integers.

        Args:
            x: An integer

        Returns:
            Two integers.
        """
        factors = self._prime_powers(x)
        if len(factors) % 2 == 0:
            i = int(len(factors) / 2)
            return factors[i], factors[i - 1]
        i = len(factors) // 2
        return factors[i], factors[i]

    def plot_output(self, name, output):
        """Plots output tensor

        Args:
            output: The output tensor
            name: Name of the output tensor, as figure name
        """
        num_filters = output.shape[3]
        grid_r, grid_c = self._get_grid_dim(num_filters)
        fig, axes = plt.subplots(nrows=min([grid_r, grid_c]),
                                 ncols=max([grid_r, grid_c]),
                                 figsize=(4 * grid_r, 4 * grid_c))

        w_min = np.min(output)
        w_max = np.max(output)
        filters = range(num_filters)

        # iterate filters
        if num_filters == 1:
            img = output[0, :, :, filters[0]]
            axes.imshow(img,
                        vmin=w_min,
                        vmax=w_max,
                        interpolation='bicubic',
                        cmap=cm.hot)
            # remove any labels from the axes
            axes.set_xticks([])
            axes.set_yticks([])
        else:
            for l, ax in enumerate(axes.flat):
                # get a single image
                img = output[0, :, :, filters[l]]
                # put it on the grid
                ax.imshow(img,
                          vmin=w_min,
                          vmax=w_max,
                          interpolation='bicubic',
                          cmap=cm.hot)
                # remove any labels from the axes
                ax.set_xticks([])
                ax.set_yticks([])

        # save figure
        img_path = os.path.join(self._output_dir, '{}.png'.format(name))
        plt.savefig(img_path, bbox_inches='tight')
        print 'finished...  {}'.format(img_path)
        plt.cla()
        plt.close("all")

    def plot_conv_outputs(self, inputs):
        """ Plot outputs of all convolution layers

        Args:
            inputs: Input image, a np array
        """
        conv_outputs = self._loader.run_conv_tensors(inputs)
        if conv_outputs is None:
            return
        for name, output in conv_outputs:
            self.plot_output(name, output)
