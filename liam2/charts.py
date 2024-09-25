import os
import math
import sys

import numpy as np
import larray as la

from liam2 import config
from liam2.expr import FunctionExpr
from liam2.utils import get_axes, ExceptionOnGetAttr, ndim, FileProducer

try:
    import matplotlib

    matplotlib.use('QtAgg')

    import matplotlib.pyplot as plt
    # set interactive mode
    # plt.ion()
except ImportError as e:
    msg = f"charts functionality is not available because 'matplotlib.pyplot' could not be imported ({e})."
    print("Warning:", msg)
    if not config.debug:
        e = ImportError(msg).with_traceback(sys.exc_info()[2])
    plt = ExceptionOnGetAttr(e)


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
        # compute a range which includes the end (1.0)
        if n == 1:
            ratios = [0.0]
        else:
            ratios = [float(i) / (n - 1) for i in range(n)]
        # shrink to [0.2, 0.7]
        ratios = [0.2 + r * 0.5 for r in ratios]
        # start from end
        ratios = [1.0 - r for r in ratios]

        cmap = plt.get_cmap('OrRd')
        return [cmap(f) for f in ratios]

    def prepare(self, args, **kwargs):
        ndim_req = self.ndim_req
        dimerror = False
        if self.check_length and len(args) > 1:
            if all(np.isscalar(a) for a in args):
                args = [np.asarray(args)]
            else:
                length = len(args[0])
                if any(len(a) != length for a in args):
                    raise ValueError("when plotting multiple arrays, they must "
                                     "have compatible axes")
        data = None
        if len(args) == 1:
            data = args[0]
            if not isinstance(data, (np.ndarray, la.Array)):
                # If data is a tuple of array/Array, it will be converted to
                # a 2d array if all arrays are the same length but to an array of arrays otherwise
                data = np.asarray(data)

            if ndim(data) == ndim_req:
                # move the last axis first so that the last dimension is stacked
                axes = list(range(data.ndim))
                data = data.transpose(axes[-1], *axes[:-1])
            elif ndim(data) == ndim_req - 1:
                if isinstance(data, la.Array):
                    # add dummy axis and move it as the first axis
                    dummy = la.Axis(1, '')
                    data = data.expand(dummy).transpose(dummy)
                else:
                    data = data[np.newaxis]
            else:
                dimerror = True
        elif all(ndim(a) == ndim_req - 1 for a in args):
            data = args
        else:
            dimerror = True
        if dimerror:
            raise ValueError(f"{self.funcname}() only works on {ndim_req - 1} or {ndim_req} dimensional data")
        return data, get_axes(data)

    def compute(self, context, *args, colors=None, grid=None, maxticks=None,
                xmin=None, xmax=None, ymin=None, ymax=None,
                fname=None, suffix='', **kwargs):
        entity = context.entity
        period = context.period
        fig = plt.figure()
        ax = fig.add_subplot(projection=self.projection)

        data, axes = self.prepare(args, **kwargs)
        if colors is None:
            colors = self.get_colors(len(axes[0]))
        fname_pattern = self._get_fname(fname, suffix)
        if grid is None:
            grid = self.show_grid
        if maxticks is None:
            maxticks = self.maxticks
        if self.show_legend:
            self.set_legend(axes[0], colors)
            axes = axes[1:]
        if self.show_axes:
            self.set_axes(ax, axes, maxticks)
        self._draw(data, colors, **kwargs)
        if self.show_axes:
            # setting x/ylim need to happen after draw, so that the "keep
            # last value" behavior of setting them to None works, otherwise
            # it breaks awfully (eg sets ylim to 0, 1)
            ax.set_xlim(left=xmin, right=xmax, emit=False)
            ax.set_ylim(bottom=ymin, top=ymax, emit=False)

        ax.grid(grid)
        if fname is None:
            plt.show()
        else:
            root, exts = os.path.splitext(fname_pattern)
            exts = exts.split('&')
            # the first extension already contains a ".", but not the others
            exts = [exts[0]] + ['.' + ext for ext in exts[1:]]
            for ext in exts:
                fname = (root + ext).format(entity=entity.name, period=period)
                print(f"writing to {fname} ...", end=' ')
                plt.savefig(config.output_directory / fname)

        # explicit close is needed for Qt backend
        plt.close(fig)

    def _draw(self, data, colors, **kwargs):
        raise NotImplementedError()

    def _set_axis_method(name):
        def set_axis(self, ax, axis, maxticks=20):
            numvalues = len(axis)
            numticks = min(maxticks, numvalues)
            step = int(math.ceil(numvalues / float(numticks)))

            set_axis_ticks = getattr(ax, f'set_{name}ticks')
            set_axis_ticks(np.arange(0, numvalues, step))
            if axis.name is not None:
                set_axis_label = getattr(ax, f'set_{name}label')
                set_axis_label(axis.name)
            set_axis_ticklabels = getattr(ax, f'set_{name}ticklabels')
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

    def set_axes(self, ax, axes, maxticks=20):
        ndim = len(axes)
        self.set_xaxis(ax, axes[0], maxticks)
        if ndim > 1:
            self.set_yaxis(ax, axes[1], maxticks)
        if ndim > 2:
            self.set_zaxis(ax, axes[2], maxticks)


class BoxPlot(Chart):
    show_legend = False
    # boxplot works fine with several arrays of different lengths
    check_length = False

    def prepare(self, args, **kwargs):
        if len(args) > 1:
            args = (args,)
        data, axes = super(BoxPlot, self).prepare(args, **kwargs)
        self.label_axis = axes[0]
        return data, axes

    def _draw(self, data, colors, **kwargs):
        axis = self.label_axis
        # tell matplotlib we don't want it to add default ticks (1 .. N)
        if not axis.iswildcard:
            kwargs['positions'] = np.arange(len(axis))
            kwargs['manage_ticks'] = False

        # boxplot does not support varargs, so if we want several boxes,
        # we must pass a tuple instead of unpacking it (ie. no * on data)
        plt.boxplot(data, **kwargs)


class Scatter(Chart):
    # our code does not handle nicely axes with floating point ticks and
    # mpl handles them fine
    show_axes = False
    colorbar_threshold = 10

    def prepare(self, args, c='b', **kwargs):
        axes = [la.Axis(np.unique(arg)) for arg in args]
        unq_colors = np.unique(c)
        if len(unq_colors) >= self.colorbar_threshold:
            # we will add a colorbar in this case, so we do not need a legend
            self.show_legend = False
        else:
            # prepend a fake axis that will be used to make a legend
            axes = [la.Axis(unq_colors)] + axes
        return args, axes

    def _draw(self, data, colors, *, cmap=None, r=None, s=None, **kwargs):
        from matplotlib.colors import ListedColormap

        if cmap is None:
            cmap = ListedColormap(colors)
        if r is not None:
            if s is not None:
                raise Exception('cannot specify both r and s arguments to '
                                'scatter')
            s = np.pi * np.asarray(r) ** 2
        sc = plt.scatter(*data, cmap=cmap, s=s, **kwargs)
        if len(colors) >= self.colorbar_threshold:
            plt.colorbar(sc)


class Plot(Chart):
    show_grid = False

    def __init__(self, *args, **kwargs):
        Chart.__init__(self, *args, **kwargs)
        self.styles = None

    def prepare(self, args, styles=None, **kwargs):
        # "inline" styles have priority over kwarg styles
        if len(args) > 1:
            # every odd is a string => we have styles, yeah !
            if all(isinstance(a, str) for a in args[1::2]):
                styles = args[1::2]
                args = args[::2]
        self.styles = styles
        return super(Plot, self).prepare(args, **kwargs)

    def _draw(self, data, colors, **kwargs):
        data = np.asarray(data)
        if self.styles is None:
            for array, color in zip(data, colors):
                kw = dict(color=color)
                kw.update(kwargs)
                plt.plot(array, **kw)
        else:
            for array, style, color in zip(data, self.styles, colors):
                kw = dict(color=color)
                kw.update(kwargs)
                plt.plot(array, style, **kw)


class StackPlot(Chart):
    def _draw(self, data, colors, **kwargs):
        data = np.asarray(data)
        x = np.arange(len(data[0]))
        plt.stackplot(x, data, colors=colors, **kwargs)


class Bar(Chart):
    show_grid = False

    def _draw(self, data, colors, *, x=None, color=None, **kwargs):
        data = np.asarray(data)
        numvalues = len(data[0])

        if x is None:
            x = np.arange(numvalues)
        # use an explicit align='center' because this is only the default for matplotlib >= 2.0
        kw = dict(width=0.5, align='center')
        kw.update(kwargs)
        # we need to handle bottom explicitly to stack several rows
        bottom = np.zeros(numvalues, dtype=data[0].dtype)
        for row, cycle_color in zip(data, colors):
            row_color = color if color is not None else cycle_color
            plt.bar(x, height=row, bottom=bottom, color=row_color, **kw)
            bottom += row


class BarH(Bar):
    show_grid = False

    def _draw(self, data, colors, *, y=None, color=None, **kwargs):
        data = np.asarray(data)
        numvalues = len(data[0])

        if y is None:
            y = np.arange(numvalues)
        # use an explicit align='center' because this is only the default for matplotlib >= 2.0
        kw = dict(height=0.5, align='center')
        kw.update(kwargs)
        # we need to handle left explicitly to stack several rows
        left = np.zeros(numvalues, dtype=data.dtype)
        for row, cycle_color in zip(data, colors):
            row_color = color if color is not None else cycle_color
            plt.barh(y, width=row, left=left, color=row_color, **kw)
            left += row


class Pie(Chart):
    show_axes = False
    show_legend = False
    ndim_req = 1

    def _draw(self, data, colors, *, title=None, **kwargs):
        if isinstance(data, la.Array):
            labels = data.axes[0].labels
            title = data.axes[0].name
            data = np.asarray(data)
        else:
            labels = None
            title = None

        kw = dict(labels=labels, colors=self.get_colors(len(data)),
                  autopct='%1.1f%%', startangle=90, title=title)
        kw.update(kwargs)
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
