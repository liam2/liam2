from __future__ import print_function

import math

import numpy as np

from expr import Expr, expr_eval
from process import Process
from utils import Axis, LabeledArray, DelayedImportModule, aslabeledarray

plt = DelayedImportModule('matplotlib.pyplot')


class Chart(Process):
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
        plt.figure()
        args = [expr_eval(arg, context) for arg in self.args]
        kwargs = dict((k, expr_eval(v, context))
                      for k, v in self.kwargs.iteritems())
        self._draw(*args, **kwargs)
        plt.show()

    def _draw(self, *args, **kwargs):
        raise NotImplementedError()

    def set_axis_method(name):
        def set_axis(self, axis, maxticks=20):
            ax = plt.gca()
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

    def set_axes(self, data, maxticks=20, skip_axes=0):
        array = aslabeledarray(data)
        axes = array.axes
        if skip_axes:
            axes = axes[skip_axes:]
        ndim = len(axes)
        self.set_xaxis(axes[0], maxticks)
        if ndim > 1:
            self.set_yaxis(axes[1], maxticks)
        if ndim > 2:
            self.set_zaxis(axes[2], maxticks)

    def set_axes_and_legend(self, data, maxticks=20, stackthreshold=2):
        array = aslabeledarray(data)
        if array.ndim >= stackthreshold:
            colors = self.get_colors(len(array))
            self.set_legend(array.axes[0], colors)
            skip_axes = 1
        else:
            colors = self.get_colors(1)
            skip_axes = 0
        self.set_axes(array, maxticks, skip_axes=skip_axes)
        return colors


class BoxPlot(Chart):
    def _draw(self, *args, **kwargs):
        plt.boxplot(*args, **self.kwargs)
        if len(args) == 1:
            args = args[0]
        self.set_axes(args)


class Plot(Chart):
    def _draw(self, *args, **kwargs):
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

        #TODO: make maxticks as an argument
        x = np.arange(len(args[0])) + 1
        colors = self.set_axes_and_legend(args, maxticks=10)
        for array, color in zip(args, colors):
            # use np.asarray to work around missing "newaxis" implementation
            # in LabeledArray
            plt.plot(x, np.asarray(array), color=color, **self.kwargs)


class StackPlot(Chart):
    def _draw(self, *args, **kwargs):
        if len(args) == 1 and args[0].ndim == 2:
            args = args[0]
            length = args.shape[-1]
        elif all(a.ndim == 1 for a in args):
            length = len(args[0])
        else:
            raise ValueError("stackplot only works with a 2D array (MxN) or "
                             "M 1D arrays (each of dimension 1xN)")

        x = np.arange(length) + 1
        colors = self.set_axes_and_legend(args)
        # use np.asarray to work around missing "newaxis" implementation
        # in LabeledArray
        plt.stackplot(x, np.asarray(args), colors=colors, **self.kwargs)


class BarChart(Chart):
    def _draw(self, *args, **kwargs):
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
        colors = self.set_axes_and_legend(data)
        numvalues = data.shape[1]

        # plots with the left of the first bar in a negative position look
        # ugly, so we shift ticks by 1 and left coordinates of bars by 0.75
        x = np.arange(numvalues) + 0.75
        bottom = np.zeros(numvalues, dtype=data.dtype)
        for row, color in zip(data, colors):
            plt.bar(left=x, height=row, width=0.5, bottom=bottom,
                    color=color, **self.kwargs)
            bottom += row


class BarChart3D(Chart):
    def _draw(self, *args, **kwargs):
        from mpl_toolkits.mplot3d import Axes3D

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

        ax = plt.gca(projection='3d')
        colors = self.set_axes_and_legend(data, maxticks=10, stackthreshold=3)
        size = xlen * ylen
        positions = np.mgrid[:xlen, :ylen] + 0.75
        xpos, ypos = positions.reshape(2, size)
        zpos = np.zeros(size)
        for array, color in zip(data, colors):
            dz = array.flatten()
            ax.bar3d(xpos, ypos, zpos, dx=0.5, dy=0.5, dz=dz, color=color)
            zpos += dz


class PieChart(Chart):
    def _draw(self, *args, **kwargs):
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
