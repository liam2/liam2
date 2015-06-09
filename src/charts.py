# encoding: utf-8
from __future__ import print_function, division

import os
import math

import numpy as np

import config
from expr import FunctionExpr
from utils import (LabeledArray, aslabeledarray, ExceptionOnGetAttr, ndim,
                   Axis, FileProducer)

try:
    import matplotlib
    matplotlib.use('Qt4Agg')
    del matplotlib
    import matplotlib.pyplot as plt
    # set interactive mode
    # plt.ion()
except ImportError, e:
    plt = ExceptionOnGetAttr(e)
    print("Warning: charts functionality is not available because "
          "'matplotlib.pyplot' could not be imported (%s)." % e)


class Chart(FunctionExpr, FileProducer):
    ext = '.png'
    show_grid = False
    show_axes = True
    show_legend = True
    maxticks = 20
    projection = None
    ndim_req = 2
    check_length = True

    def get_colors(self, n):
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

    def prepare(self, args, kwargs):
        ndim_req = self.ndim_req
        dimerror = ValueError("%s() only works on %d or %d dimensional data"
                              % (self.funcname, ndim_req - 1, ndim_req))
        if self.check_length and len(args) > 1:
            if all(np.isscalar(a) for a in args):
                args = [np.asarray(args)]
            else:
                length = len(args[0])
                if any(len(a) != length for a in args):
                    raise ValueError("when plotting multiple arrays, they must "
                                     "have compatible axes")
        if len(args) == 1:
            data = args[0]
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)
            if ndim(data) == ndim_req:
                # move the last axis first so that the last dimension is stacked
                axes = range(data.ndim)
                data = data.transpose(axes[-1], *axes[:-1])
            elif ndim(data) == ndim_req - 1:
                if isinstance(data, LabeledArray):
                    # TODO: implement np.newaxis in LabeledArray.__getitem__
                    data = LabeledArray(np.asarray(data)[np.newaxis],
                                        dim_names=['dummy'] + data.dim_names,
                                        pvalues=[[0]] + data.pvalues)
                else:
                    data = data[np.newaxis]
            else:
                raise dimerror
        elif all(ndim(a) == ndim_req - 1 for a in args):
            data = args
        else:
            raise dimerror
        return data, aslabeledarray(data).axes

    def compute(self, context, *args, **kwargs):
        entity = context.entity
        period = context.period

        fig = plt.figure()

        data, axes = self.prepare(args, kwargs)
        colors = kwargs.pop('colors', None)
        if colors is None:
            colors = self.get_colors(len(axes[0]))
        fname = self._get_fname(kwargs)
        grid = kwargs.pop('grid', self.show_grid)
        maxticks = kwargs.pop('maxticks', self.maxticks)
        projection = self.projection

        if self.show_legend:
            self.set_legend(axes[0], colors)
            axes = axes[1:]
        if self.show_axes:
            self.set_axes(axes, maxticks, projection)

        self._draw(data, colors, **kwargs)

        plt.grid(grid)
        if fname is None:
            plt.show()
        else:
            root, exts = os.path.splitext(fname)
            exts = exts.split('&')
            # the first extension already contains a ".", but not the others
            exts = [exts[0]] + ['.' + ext for ext in exts[1:]]
            for ext in exts:
                fname = (root + ext).format(entity=entity.name, period=period)
                print("writing to", fname, "...", end=' ')
                plt.savefig(os.path.join(config.output_directory, fname))

        # explicit close is needed for Qt4 backend
        plt.close(fig)

    def _draw(self, data, colors, **kwargs):
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

    def set_legend(self, axis, colors):
        # we don't want a legend when there is only one item
        if len(axis) < 2:
            return
        proxies = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in colors]
        plt.legend(proxies, axis.labels, title=axis.name)

    def set_axes(self, axes, maxticks=20, projection=None):
        ndim = len(axes)
        self.set_xaxis(axes[0], maxticks, projection)
        if ndim > 1:
            self.set_yaxis(axes[1], maxticks, projection)
        if ndim > 2:
            self.set_zaxis(axes[2], maxticks, projection)


class BoxPlot(Chart):
    show_legend = False
    # boxplot works fine with several arrays of different lengths
    check_length = False

    def _draw(self, data, colors, **kwargs):
        # boxplot does not support varargs, so if we want several boxes,
        # we must pass a tuple instead of unpacking it (ie. no * on data)
        plt.boxplot(data, **kwargs)


class Scatter(Chart):
    # our code does not handle nicely axes with floating point ticks and
    # mpl handles them fine
    show_axes = False
    colorbar_threshold = 10

    def prepare(self, args, kwargs):
        axes = [Axis(None, np.unique(arg)) for arg in args]
        c = kwargs.get('c', 'b')
        unq_colors = np.unique(c)
        if len(unq_colors) >= self.colorbar_threshold:
            # we will add a colorbar in this case, so we do not need a legend
            self.show_legend = False
        else:
            # prepend a fake axis that will be used to make a legend
            axes = [Axis(None, unq_colors)] + axes
        return args, axes

    def _draw(self, data, colors, **kwargs):
        from matplotlib.colors import ListedColormap
        if 'cmap' not in kwargs:
            kwargs['cmap'] = ListedColormap(colors)
        r = kwargs.pop('r', None)
        if r is not None:
            if kwargs.get('s') is not None:
                raise Exception('cannot specify both r and s arguments to '
                                'scatter')
            kwargs['s'] = np.pi * np.asarray(r) ** 2
        sc = plt.scatter(*data, **kwargs)
        if len(colors) >= self.colorbar_threshold:
            plt.colorbar(sc)


class Plot(Chart):
    show_grid = True

    def __init__(self, *args, **kwargs):
        Chart.__init__(self, *args, **kwargs)
        self.styles = None

    def prepare(self, args, kwargs):
        # "inline" styles have priority over kwarg styles
        styles = kwargs.pop('styles', None)
        if len(args) > 1:
            # every odd is a string => we have styles, yeah !
            if all(isinstance(a, basestring) for a in args[1::2]):
                styles = args[1::2]
                args = args[::2]
        self.styles = styles
        return super(Plot, self).prepare(args, kwargs)

    def _draw(self, data, colors, **kwargs):
        x = np.arange(len(data[0])) + 1

        # we use np.asarray to work around missing "newaxis" implementation
        # in LabeledArray
        if self.styles is None:
            for array, color in zip(data, colors):
                kw = dict(color=color)
                kw.update(kwargs)
                plt.plot(x, np.asarray(array), **kw)
        else:
            for array, style, color in zip(data, self.styles, colors):
                kw = dict(color=color)
                kw.update(kwargs)
                plt.plot(x, np.asarray(array), style, **kw)


class StackPlot(Chart):
    def _draw(self, data, colors, **kwargs):
        x = np.arange(len(data[0])) + 1
        # use np.asarray to work around missing "newaxis" implementation
        # in LabeledArray
        plt.stackplot(x, np.asarray(data), colors=colors, **kwargs)


class Bar(Chart):
    show_grid = True

    def _draw(self, data, colors, **kwargs):
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
    show_grid = True

    def _draw(self, data, colors, **kwargs):
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


class Pie(Chart):
    show_axes = False
    show_legend = False
    ndim_req = 1

    def _draw(self, data, colors, **kwargs):
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
    'scatter': Scatter,
    'boxplot': BoxPlot,
    'plot': Plot,
    'stackplot': StackPlot,
    'bar': Bar,
    'barh': BarH,
    'pie': Pie,
}
