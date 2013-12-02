from __future__ import print_function

import math

import numpy as np

from expr import Expr, expr_eval
from process import Process
from utils import LabeledArray, aslabeledarray, ExceptionOnGetAttr

try:
    import matplotlib
    matplotlib.use('Qt4Agg')
    del matplotlib
    import matplotlib.pyplot as plt
except ImportError, e:
    plt = ExceptionOnGetAttr(e)
    print("Warning: charts functionality is not available because "
          "'matplotlib.pyplot' could not be imported (%s)." % e)

try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError, e:
    if not isinstance(plt, ExceptionOnGetAttr):
        print("Warning: 3D charts are not available because "
              "'mpl_toolkits.mplot3d.AxesED' could not be imported (%s).")


class Chart(Process):
    axes = True
    legend = True
    maxticks = 20
    projection = None
    stackthreshold = 2

    def __init__(self, *args, **kwargs):
        Process.__init__(self)
        self.args = args
        self.kwargs = kwargs

    def expressions(self):
        for arg in self.args:
            if isinstance(arg, Expr):
                yield arg

    def get_colors(self, n):
        if n == 1:
            return ['#ff4422']
        else:
            from matplotlib import cm
            cmap = cm.get_cmap('OrRd')
            return [cmap(float(i) / n) for i in range(n)]

    def run(self, context):
        fig = plt.figure()
        args = [expr_eval(arg, context) for arg in self.args]
        kwargs = dict((k, expr_eval(v, context))
                      for k, v in self.kwargs.iteritems())

        maxticks = kwargs.pop('maxticks', self.maxticks)
        projection = self.projection
        stackthreshold = self.stackthreshold
        if self.legend and self.axes:
            colors = self.set_axes_and_legend(args, maxticks=maxticks,
                                              stackthreshold=stackthreshold,
                                              projection=projection,
                                              kwargs=kwargs)
        elif self.axes:
            self.set_axes(args, maxticks=maxticks, projection=projection,
                          kwargs=kwargs)
            colors = None
        else:
            colors = None
        self._draw(colors, *args, **kwargs)
        plt.show()
        # explicit close is needed for Qt4 backend
        plt.close(fig)

    def _draw(self, colors, *args, **kwargs):
        raise NotImplementedError()

    def set_axis_method(name):
        def set_axis(self, axis, maxticks=20, projection=None):
            ax = plt.gca(projection=projection)
            numvalues = len(axis)
            numticks = min(maxticks, numvalues)
            step = int(math.ceil(numvalues / float(numticks)))
            set_axis_ticks = getattr(ax, 'set_%sticks' % name)
            set_axis_label = getattr(ax, 'set_%slabel' % name)
            set_axis_ticklabels = getattr(ax, 'set_%sticklabels' % name)
            set_axis_ticks(np.arange(1, numvalues + 1, step))
            if axis.name is not None:
                set_axis_label(axis.name)
            set_axis_ticklabels(axis.labels[::step])
        return set_axis
    set_xaxis = set_axis_method('x')
    set_yaxis = set_axis_method('y')
    set_zaxis = set_axis_method('z')

    def set_legend(self, axis, colors, **kwargs):
        # we don't want a legend when there is only one item
        if len(axis) < 2:
            return
        proxies = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in colors]
        plt.legend(proxies, axis.labels, title=axis.name, **kwargs)

    def set_axes(self, data, maxticks=20, skip_axes=0, projection=None,
                 kwargs=None):
        if len(data) == 1:
            data = data[0]
        array = aslabeledarray(data)
        axes = array.axes
        if skip_axes:
            axes = axes[skip_axes:]
        ndim = len(axes)
        self.set_xaxis(axes[0], maxticks, projection)
        if ndim > 1:
            self.set_yaxis(axes[1], maxticks, projection)
        if ndim > 2:
            self.set_zaxis(axes[2], maxticks, projection)

    def set_axes_and_legend(self, data, maxticks=20, stackthreshold=2,
                            projection=None, kwargs=None):
        if len(data) == 1:
            data = data[0]
        array = aslabeledarray(data)
        if array.ndim >= stackthreshold:
            colors = self.get_colors(len(array))
            self.set_legend(array.axes[0], colors)
            skip_axes = 1
        else:
            colors = self.get_colors(1)
            skip_axes = 0
        self.set_axes(array, maxticks, skip_axes, projection, kwargs)
        return colors


class BoxPlot(Chart):
    legend = False

    def _draw(self, colors, *args, **kwargs):
        plt.boxplot(*args, **kwargs)


class Plot(Chart):
    def _draw(self, colors, *args, **kwargs):
        if (len(args) == 1 and isinstance(args[0], np.ndarray) and
                args[0].ndim == 2):
            args = args[0]

        if any(isinstance(a, np.ndarray) and a.ndim > 1 for a in args):
            raise ValueError("too many dimensions to plot")

        if not isinstance(args, np.ndarray):
            length = len(args[0])
            if any(len(a) != length for a in args):
                raise ValueError("when plotting multiple arrays, they should "
                                 "have compatible axes")

        x = np.arange(len(args[0])) + 1
        for array, color in zip(args, colors):
            # use np.asarray to work around missing "newaxis" implementation
            # in LabeledArray
            plt.plot(x, np.asarray(array), color=color, **kwargs)


class StackPlot(Chart):
    def _draw(self, colors, *args, **kwargs):
        if len(args) == 1 and args[0].ndim == 2:
            args = args[0]
            length = args.shape[-1]
        elif all(a.ndim == 1 for a in args):
            length = len(args[0])
        else:
            raise ValueError("stackplot only works with a 2D array (MxN) or "
                             "M 1D arrays (each of dimension 1xN)")

        x = np.arange(length) + 1
        # use np.asarray to work around missing "newaxis" implementation
        # in LabeledArray
        plt.stackplot(x, np.asarray(args), colors=colors, **kwargs)


class BarChart(Chart):
    def _draw(self, colors, *args, **kwargs):
        data = args[0]
        if data.ndim == 1:
            if isinstance(data, LabeledArray):
                #TODO: implement np.newaxis in LabeledArray.__getitem__
                data = LabeledArray(np.asarray(data)[np.newaxis, :],
                                    dim_names=['dummy'] + data.dim_names,
                                    pvalues=[[0]] + data.pvalues)
            else:
                data = data[np.newaxis, :]
        elif data.ndim != 2:
            raise ValueError("barchart only works on 1 or 2 dimensional data")

        plt.grid(True)
        numvalues = data.shape[1]

        # plots with the left of the first bar in a negative position look
        # ugly, so we shift ticks by 1 and left coordinates of bars by 0.75
        x = np.arange(numvalues) + 0.75
        bottom = np.zeros(numvalues, dtype=data.dtype)
        for row, color in zip(data, colors):
            plt.bar(left=x, height=row, width=0.5, bottom=bottom,
                    color=color, **kwargs)
            bottom += row


class BarChart3D(Chart):
    maxticks = 10
    projection = '3d'
    stackthreshold = 3

    def _draw(self, colors, *args, **kwargs):

        data = args[0]
        if data.ndim == 2:
            if isinstance(data, LabeledArray):
                #TODO: implement np.newaxis in LabeledArray.__getitem__
                data = LabeledArray(np.asarray(data)[np.newaxis, :, :],
                                    dim_names=['dummy'] + data.dim_names,
                                    pvalues=[[0]] + data.pvalues)
            else:
                data = data[np.newaxis, :, :]
        elif data.ndim != 3:
            raise ValueError("barchart3d only works on 2 and 3 dimensional "
                             "data")

        _, xlen, ylen = data.shape

        kw = dict(dx=0.5, dy=0.5)
        kw.update(kwargs)
        ax = plt.gca(projection='3d')
        size = xlen * ylen
        positions = np.mgrid[:xlen, :ylen] + 1.0
        xpos, ypos = positions.reshape(2, size)
        xpos -= kw['dx'] / 2
        ypos -= kw['dy'] / 2
        zpos = np.zeros(size)
        for array, color in zip(data, colors):
            dz = array.flatten()
            ax.bar3d(xpos, ypos, zpos, dz=dz, color=color,
                     **kw)
            zpos += dz


class PieChart(Chart):
    axes = False
    legend = False

    def _draw(self, colors, *args, **kwargs):
        data = args[0]
        if data.ndim != 1:
            raise ValueError("piechart only works on 1 dimensional data")

        if isinstance(data, LabeledArray) and data.pvalues:
            labels = data.pvalues[0]
            plt.title(data.dim_names[0])
        else:
            labels = None
        kw = dict(labels=labels, colors=self.get_colors(len(data)),
                  autopct='%1.1f%%', shadow=True, startangle=90)
        kw.update(kwargs)
        plt.pie(data, **kw)
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')


functions = {
    'boxplot': BoxPlot,
    'plot': Plot,
    'stackplot': StackPlot,
    'barchart': BarChart,
    'barchart3d': BarChart3D,
    'piechart': PieChart,
}
