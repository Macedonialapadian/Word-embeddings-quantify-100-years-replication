import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from math import sqrt

SPINE_COLOR = "gray"


def latexify(fig_width=None, fig_height=None, columns=1):
    """
    Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float, optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    assert columns in [1, 2]

    # Width and max height in inches (IEEE-style)
    if fig_width is None:
        fig_width = 6.9 if columns == 1 else 13.8  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 16.0
    if fig_height > MAX_HEIGHT_INCHES:
        print(
            f"WARNING: fig_height too large: {fig_height}, "
            f"reducing to {MAX_HEIGHT_INCHES} inches."
        )
        fig_height = MAX_HEIGHT_INCHES

    params = {
        "backend": "ps",
        # "text.latex.preamble": [r"\usepackage{gensymb}"],
        "axes.labelsize": 15,
        "axes.titlesize": 5,
        "font.size": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": False,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
        "lines.linewidth": 0.4,
        "lines.markersize": 0.4,
        "xtick.major.pad": 1,
        "ytick.major.pad": 0,
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction="out", color=SPINE_COLOR)

    return ax
