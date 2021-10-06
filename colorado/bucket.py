import numpy
from typing import Sequence
import plotly.graph_objects as go
from .aims_tools import bucket_aims_to_ndarray
import logging
log = logging.getLogger(__name__)


def get_bucket_g_o(bucket, name=None, shift=(0, 0, 0), **kwargs):

    if bucket.shape[1] != 3:
        raise ValueError(
            "The shape array is not correct: expected (N,3), got{}".format(bucket.shape))

    bucket = bucket+shift
    x, y, z = bucket.T

    # set default marker properties from otional arguments
    marker = kwargs.get("marker", dict())
    opacity = marker.get('opacity', 1)
    if opacity < 1:
        log.warning("Opacity < 1 is buggy in Plotly 3D Scatter plot")

    marker['opacity'] = opacity
    marker['size'] = marker.get('size', 1)

    s3d = go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=marker,
        name=name,
    )

    return s3d


def get_aims_bucket_g_o(aims_bucket, name=None, shift=(0, 0, 0), **kwargs):
    """Plot a soma.aims Bucket"""
    bucket = bucket_aims_to_ndarray(aims_bucket) + shift
    return get_bucket_g_o(bucket, name=name, **kwargs)


def get_aims_bucket_map_g_o(aims_bucket_map, name=None, shift=(0, 0, 0), **kwargs):
    """Draw a bucketMap object, which is obtained with aims.read() on .bck files"""
    buckets_g_o = list()
    for aims_bucket in aims_bucket_map:
        buckets_g_o.append(get_aims_bucket_g_o(
            aims_bucket, name=name, shift=(0, 0, 0), **kwargs))

    return buckets_g_o


def draw_numpy_bucket(bucket, fig=None):
    """Draw one bucket from numpy array

    Args:
        bucket (numpy.ndarray, shape (N,3)): bucket

    Returns:
        plotly.graphic_objects.Figure: a plotly figure representing the bucket
    """
    assert bucket.shape[1] == 3,\
        "wrong shape: expected (N,3) got {}".format(bucket.shape)
    x, y, z = bucket.T
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(x=x, y=y, z=z, mode='markers',
                     marker=dict(size=1, opacity=1)
                     )
    )
    return fig


def draw_numpy_buckets(buckets, labels=None,
                       transpose=True, shift=(0, 0, 0), fig=None):
    """Draw buckets from numpy arrays.

    :param buckets: a list of arrays representing the buckets to plot.
        Each array must be a Nx3 numpy.ndarray containing the coordinates of N points
    :type buckets: Sequence[numpy.ndarray]
    :param transpose: for plot reasons, transpose each bucket array if True, defaults to True
    :type transpose: bool, optional
    :param x_shift: Amount of x shift to add to each sulcus, defaults to 0
    :type x_shift: num, optional

    """

    gos = []

    if not isinstance(buckets, dict):
        if labels is None:
            labels = range(len(buckets))
        buckets = dict(zip(labels, buckets))

    if fig is None:
        fig = go.Figure()

    shift = numpy.array(shift)

    for i, (name, sulcus) in enumerate(buckets.items()):
        sulcus = sulcus+shift*i

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
            go.Scatter3d(x=x, y=y, z=z, mode='markers',
                         marker=dict(size=1, opacity=1),
                         name=name)
        )

    for g in gos:
        fig.add_trace(g)

    # markers in the legend are of fixed (big) size
    fig.update_layout(legend={'itemsizing': 'constant'})

    return fig
