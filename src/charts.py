from __future__ import print_function

import math

import numpy as np

from expr import Expr, expr_eval
from process import Process
from utils import LabeledArray


class Chart(Process):
    def __init__(self, expr, **kwargs):
        Process.__init__(self)
        self.expr = expr
        self.kwargs = kwargs

    def expressions(self):
        if isinstance(self.expr, Expr):
            yield self.expr

    def get_colors(self, n):
        if n == 1:
            return ['#ff4422']
        else:
            from matplotlib import cm
            cmap = cm.get_cmap('OrRd')
            return [cmap(float(i) / n) for i in range(n)]


class BarChart(Chart):
    def run(self, context):
        import matplotlib.pyplot as plt

        data = expr_eval(self.expr, context)
        if data.ndim == 1:
            if isinstance(data, LabeledArray):
                #TODO: override LabeledArray.reshape
                data = LabeledArray(data.reshape(1, len(data)),
                                    dim_names=['dummy'] + data.dim_names,
                                    pvalues=[[0]] + data.pvalues)
            else:
                data = data.reshape(1, len(data))
        elif data.ndim != 2:
            raise ValueError("barchart only works on 1 or 2 dimensional data")

        numbars, numvalues = data.shape
        colors = self.get_colors(numbars)

        # start a new plot
        _, ax = plt.subplots()

        # plots with the first bar start in a negative position (even via the
        # align center) look ugly, so we shift it by 1
        ind = np.arange(1, numvalues + 1)
        bottom = np.zeros(numvalues, dtype=data.dtype)
        plots = []
        for row, color in zip(data, colors):
            plot = ax.bar(left=ind, height=row, width=0.5,
                          bottom=bottom, align='center', color=color,
                          **self.kwargs)
            bottom += row
            plots.append(plot)

        # commented because str(groupby) is currently useless
        #ax.set_ylabel(str(self.expr))
        #ax.set_title('Scores by group and gender')
        numticks = min(20.0, float(numvalues))
        step = int(math.ceil(numvalues / numticks))
        ax.set_xticks(np.arange(1, numvalues + 1, step))
        if isinstance(data, LabeledArray) and data.pvalues:
            ax.set_xlabel(data.dim_names[1])
            ax.set_xticklabels(data.pvalues[1][::step])
            if len(plots) > 1:
                plt.legend([plot[0] for plot in plots], data.pvalues[0],
                           title=data.dim_names[0])
        elif len(plots) > 1:
            plt.legend([plot[0] for plot in plots], range(len(data)))

        ax.grid(True)
        plt.show()


class BarChart3D(Chart):
    def run(self, context):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        data = expr_eval(self.expr, context)
        if data.ndim == 2:
            if isinstance(data, LabeledArray):
                #TODO: override LabeledArray.reshape
                data = LabeledArray(data.reshape((1,) + data.shape),
                                    dim_names=['dummy'] + data.dim_names,
                                    pvalues=[[0]] + data.pvalues)
            else:
                data = data[np.newaxis, :, :]
        elif data.ndim != 3:
            raise ValueError("barchart3d only works on 2 and 3 dimensional "
                             "data")

        numbars, xlen, ylen = data.shape
        colors = self.get_colors(numbars)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        size = xlen * ylen
        positions = np.mgrid[:xlen, :ylen] + 0.25
        xpos, ypos = positions.reshape(2, size)
        zpos = np.zeros(size)
        for array, color in zip(data, colors):
            dz = array.flatten()
            ax.bar3d(xpos, ypos, zpos, dx=0.5, dy=0.5, dz=dz, color=color)
            zpos += dz

        if isinstance(data, LabeledArray) and data.pvalues:
            ax.set_xticks(np.arange(xlen) + 0.5)
            ax.set_xlabel(data.dim_names[1])
            ax.set_xticklabels(data.pvalues[1]) #[::step])

            ax.set_yticks(np.arange(ylen) + 0.5)
            ax.set_ylabel(data.dim_names[2])
            ax.set_yticklabels(data.pvalues[2]) #[::step])

            if len(colors) > 1:
                proxies = [plt.Rectangle((0, 0), 1, 1, fc=color)
                           for color in colors]
                ax.legend(proxies, data.pvalues[0], title=data.dim_names[0])
        plt.show()


class PieChart(Chart):
    def run(self, context):
        import matplotlib.pyplot as plt

        data = expr_eval(self.expr, context)
        if data.ndim != 1:
            raise ValueError("piechart only works on 1 dimensional data")

        if isinstance(data, LabeledArray) and data.pvalues:
            labels = data.pvalues[0]
            plt.legend(title=data.dim_names[0])
        else:
            labels = None
        plt.pie(data, labels=labels, colors=self.get_colors(len(data)),
                autopct='%1.1f%%', shadow=True, startangle=90)
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')
        plt.show()


functions = {
    'barchart': BarChart,
    'barchart3d': BarChart3D,
    'piechart': PieChart
}
