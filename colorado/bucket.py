import numpy
from typing import Sequence
import plotly.graph_objects as go
from .aims_tools import aims_bucket_to_ndarray


def get_bucket_g_o(bucket, name=None, **kwargs):

    if bucket.shape[1] != 3:
        raise ValueError(
            "The shape array is not correct: expected (N,3), got{}".format(bucket.shape))

    x, y, z = bucket.T

    return go.Scatter3d(x=x, y=y, z=z, mode='markers',
                        marker=dict(size=1, opacity=1),
                        name=name)


def get_aims_bucket_g_o(aims_bucket, name=None, shift=(0, 0, 0), **kwargs):
    """Plot a soma.aims Bucket"""
    bucket = aims_bucket_to_ndarray(aims_bucket) + shift
    return get_bucket_g_o(bucket, name=name, **kwargs)


def draw_numpy_buckets(list_of_buckets, labels=None,
                 transpose=True, x_shift=0, fig=None):
    """Draw buckets from numpy arrays.

    :param list_of_buckets: a list of arrays representing the buckets to plot.
        Each array must be a Nx3 numpy.ndarray containing the coordinates of N points
    :type list_of_buckets: Sequence[numpy.ndarray]
    :param transpose: for plot reasons, transpose each bucket array if True, defaults to True
    :type transpose: bool, optional
    :param x_shift: Amount of x shift to add to each sulcus, defaults to 0
    :type x_shift: num, optional

    """

    gos = []

    if labels is None:
        labels = [None] * len(list_of_buckets)

    assert len(labels) == len(
        list_of_buckets), "ERROR: len(labels) != len(list_of_sulci)"

    if fig is None:
        fig = go.Figure()

    for i, sulcus in enumerate(list_of_buckets):

        if transpose:
            sulcus = sulcus.T

        assert sulcus.shape[0] == 3,\
            "The shape array is not correct, first dimension should be 3, got{}".format(
                sulcus.shape[0])

        x, y, z = sulcus

        try:
            len(x)
        except:
            # x is a number and not iterable
            raise ValueError(
                "Only one point found in the sulcus, make sure you are providing a list of sulci to the plot function.")

        # do not set opacity<1 because of a plotpy issue:
        # https://github.com/plotly/plotly.js/issues/2717
        gos.append(
            go.Scatter3d(x=x+x_shift*i, y=y, z=z, mode='markers',
                         marker=dict(size=1, opacity=1),
                         name=labels[i])
        )

    for g in gos:
        fig.add_trace(g)

    # markers in the legend are of fixed (big) size
    fig.update_layout(legend={'itemsizing': 'constant'})

    return fig
