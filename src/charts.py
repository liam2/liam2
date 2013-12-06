from __future__ import print_function

import math

import numpy as np

from expr import Expr, expr_eval
from process import Process
from utils import LabeledArray, aslabeledarray, ExceptionOnGetAttr

try:
    # we do not use the qt backend because when the python script is run
    # by nppexec (in notepad++), the qt window does not open :(
    #import matplotlib
    #matplotlib.use('Qt4Agg')
    #del matplotlib
    import matplotlib.pyplot as plt
except ImportError, e:
    plt = ExceptionOnGetAttr(e)
    print("Warning: charts functionality is not available because "
          "'matplotlib.pyplot' could not be imported (%s)." % e)

try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError, e:
    Axes3D = None
    if not isinstance(plt, ExceptionOnGetAttr):
        print("Warning: 3D charts are not available because "
              "'mpl_toolkits.mplot3d.AxesED' could not be imported (%s)." % e)


def ndim(arraylike):
    n = 0
    while isinstance(arraylike, (list, tuple)):
        if not arraylike:
            raise ValueError('Cannot compute ndim of array with empty dim')
        #XXX: check that other elements have the same length?
        arraylike = arraylike[0]
        n += 1
    if isinstance(arraylike, np.ndarray):
        n += arraylike.ndim
    return n


class Chart(Process):
    grid = False
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

    def get_colors(self, n, outline=False):
        from matplotlib import cm

        # compute a range which includes the end (1.0)
        if n == 1:
            ratios = [0.0]
        else:
            ratios = [float(i) / (n - 1) for i in range(n)]
        # shrink to [0.2, 0.7]
        ratios = [0.2 + r * 0.5 for r in ratios]
        # start from end
        ratios = [1.0 - r for r in ratios]

        cmap = cm.get_cmap('OrRd')
        return [cmap(f) for f in ratios]

    def prepare(self, args, func_name, ndim_req):
        sequence = (tuple, list, np.ndarray)
        if len(args) > 1:
            if all(np.isscalar(a) for a in args):
                args = [np.asarray(args)]
            else:
                length = len(args[0])
                # this leaves the door open for mixed types like in plot()
                if any(isinstance(a, sequence) and len(a) != length
                       for a in args):
                    raise ValueError("when plotting multiple arrays, they must "
                                     "have compatible axes")
                # args = [np.asarray(args)]

        if len(args) == 1:
            data = args[0]
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)
            if data.ndim == ndim_req:
                return data
            elif data.ndim == ndim_req - 1:
                if isinstance(data, LabeledArray):
                    #TODO: implement np.newaxis in LabeledArray.__getitem__
                    return LabeledArray(np.asarray(data)[np.newaxis],
                                        dim_names=['dummy'] + data.dim_names,
                                        pvalues=[[0]] + data.pvalues)
                else:
                    return data[np.newaxis]
            else:
                raise ValueError("%s only works on %d or %d dimensional data"
                                 % (func_name, ndim_req - 1,
                                    ndim_req))
        elif all(ndim(a) == ndim_req - 1 for a in args):
            return args
        else:
            # raise ValueError("stackplot only works with a 2D array (MxN) or "
            #                  "M 1D arrays (each of dimension 1xN)")
            raise ValueError("%s only works on %d or %d dimensional data"
                             % (func_name, ndim_req - 1, ndim_req))
        # if any(isinstance(a, np.ndarray) and a.ndim > 1 for a in args):
        #     raise ValueError("too many dimensions to plot")
        #

    def run(self, context):
        fig = plt.figure()
        args = [expr_eval(arg, context) for arg in self.args]
        kwargs = dict((k, expr_eval(v, context))
                      for k, v in self.kwargs.iteritems())
        grid = kwargs.pop('grid', self.grid)

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
        plt.grid(grid)
        plt.show()
        # explicit close is needed for Qt4 backend
        plt.close(fig)

    def _draw(self, colors, *args, **kwargs):
        raise NotImplementedError()

    def _set_axis_method(name):
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
    set_xaxis = _set_axis_method('x')
    set_yaxis = _set_axis_method('y')
    set_zaxis = _set_axis_method('z')

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
        """
        kwargs are modified inplace (colors is popped)
        """
        colors = kwargs.pop('colors', None) if kwargs is not None else None
        if len(data) == 1:
            data = data[0]
        array = aslabeledarray(data)
        if array.ndim >= stackthreshold:
            if colors is None:
                colors = self.get_colors(len(array))
            self.set_legend(array.axes[0], colors)
            skip_axes = 1
        else:
            if colors is None:
                colors = self.get_colors(1)
            skip_axes = 0
        self.set_axes(array, maxticks, skip_axes, projection, kwargs)
        return colors


class BoxPlot(Chart):
    legend = False

    def _draw(self, colors, *args, **kwargs):
        # data = self.prepare(args, 'boxplot', 2)
        # plt.boxplot(data, **kwargs)
        plt.boxplot(*args, **kwargs)


class Plot(Chart):
    def _draw(self, colors, *args, **kwargs):
        data = self.prepare(args, 'plot', 2)
        x = np.arange(len(data[0])) + 1
        for array, color in zip(data, colors):
            kw = dict(color=color)
            kw.update(kwargs)

            # use np.asarray to work around missing "newaxis" implementation
            # in LabeledArray
            plt.plot(x, np.asarray(array), **kw)


class StackPlot(Chart):
    def _draw(self, colors, *args, **kwargs):
        data = self.prepare(args, 'stackplot', 2)
        x = np.arange(len(data[0])) + 1
        # use np.asarray to work around missing "newaxis" implementation
        # in LabeledArray
        plt.stackplot(x, np.asarray(data), colors=colors, **kwargs)


class Bar(Chart):
    grid = True

    def _draw(self, colors, *args, **kwargs):
        data = self.prepare(args, 'bar', 2)
        numvalues = len(data[0])

        # plots with the left of the first bar in a negative position look
        # ugly, so we shift ticks by 1 and left coordinates of bars by
        # 1 - width / 2
        width = kwargs.get('width', 0.5)
        left = np.arange(numvalues) + 1.0
        left -= width / 2
        kw = dict(left=left, width=width)
        kw.update(kwargs)
        bottom = np.zeros(numvalues, dtype=data[0].dtype)
        for row, color in zip(data, colors):
            if 'color' not in kwargs:
                kw['color'] = color
            plt.bar(height=row, bottom=bottom, **kw)
            bottom += row


class BarH(Bar):
    grid = True

    def _draw(self, colors, *args, **kwargs):
        data = self.prepare(args, 'barh', 2)
        numvalues = len(data[0])

        # plots with the bottom of the first bar in a negative position look
        # ugly, so we shift ticks by 1 and bottom coordinates of bars by
        # 1 - height / 2
        height = kwargs.get('height', 0.5)
        bottom = np.arange(numvalues) + 1.0
        bottom -= height / 2
        kw = dict(bottom=bottom, height=height)
        kw.update(kwargs)
        left = np.zeros(numvalues, dtype=data.dtype)
        for row, color in zip(data, colors):
            if 'color' not in kwargs:
                kw['color'] = color
            plt.barh(width=row, left=left, **kw)
            left += row


class Bar3D(Bar):
    maxticks = 10
    projection = '3d'
    stackthreshold = 3

    def _draw(self, colors, *args, **kwargs):
        data = self.prepare(args, 'bar3d', 3)

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
            if 'color' not in kwargs:
                kw['color'] = color
            ax.bar3d(xpos, ypos, zpos, dz=dz, **kw)
            zpos += dz


class Pie(Chart):
    axes = False
    legend = False

    def _draw(self, colors, *args, **kwargs):
        # data = self.prepare(args, 'pie', 1)
        data = args[0]
        if data.ndim != 1:
            raise ValueError("piechart only works on 1 dimensional data")

        if isinstance(data, LabeledArray) and data.pvalues:
            labels = data.pvalues[0]
            title = data.dim_names[0]
        else:
            labels = None
            title = None

        kw = dict(labels=labels, colors=self.get_colors(len(data)),
                  autopct='%1.1f%%', startangle=90, title=title)
        kw.update(kwargs)
        title = kw.pop('title', None)
        if title is not None:
            plt.title(title)
        plt.pie(data, **kw)
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')


functions = {
    'boxplot': BoxPlot,
    'plot': Plot,
    'stackplot': StackPlot,
    'bar': Bar,
    'barh': BarH,
    'bar3d': Bar3D,
    'pie': Pie,
}
