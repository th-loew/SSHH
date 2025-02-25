import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.lines import Line2D

from harper import Harper
from localization_measure import LocalizationMeasure

F = Harper.IrrationalFrequency
COLORS = [np.asarray(to_rgb(h)) for h in plt.rcParams['axes.prop_cycle'].by_key()['color']]


def flatten_axes(axes: plt.Axes | np.ndarray) -> tuple[plt.Axes, ...]:  # check: ignore
    return tuple(axes if isinstance(axes, plt.Axes) else axes.flatten())  # type: ignore[arg-type]


def get_color(instance: LocalizationMeasure.Method | None)\
        -> tuple[float, float, float]:
    if instance is None:
        return 0, 0, 0
    error = ValueError(f"Don't know color for {instance} of type {type(instance)}")
    try:
        if isinstance(instance, LocalizationMeasure.Method):
            c = COLORS[LocalizationMeasure.METHODS.index(instance)]
            return float(c[0]), float(c[1]), float(c[2])
    except (KeyError, IndexError) as e:
        raise error from e
    raise error


def get_label(key: Harper.Frequency) -> str:
    if isinstance(key, Harper.IrrationalFrequency):
        try:
            return {
                F.GOLDEN_NUMBER: r"$\frac{1 + \sqrt{5}}{2}$",
                F.GOLDEN_NUMBER_REDUCED: r"$\frac{\sqrt{5}}{2}$",
                F.SEC_MOST_IRRAT: r"$1 + \sqrt{2}$",
                F.SEC_MOST_IRRAT_RED: r"$\sqrt{2}$",
                F.PI_THIRD: r"$\frac{\pi}{3}$",
                F.SQRT_2_NORMALIZED: r"$\frac{\sqrt{2}}{1.4}$",
                F.SQRT_1_2_NORMALIZED: r"$\frac{1.4}{\sqrt{2}}$",
                F.PI: r"$\pi$",
            }[key]
        except KeyError as e:
            raise ValueError(f"Don't know label for Harper.IrrationalFrequency.{key}") from e
    if isinstance(key, Harper.Frequency):
        return f"{float(key)}"
    raise ValueError(f"Don't know label for {key} of type {type(key)}")


class HandlerLineCollectionShowAll(HandlerLineCollection):
    """
    Custom Handler for LineCollection instances.
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, _ = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = np.full_like(xdata, height / (numlines + 1))
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[1] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines