import plotly.graph_objects as go
import logging
log = logging.getLogger(__name__)


def _get_marker_kwarg_for_scatterplot(**kwargs):
    # set default marker properties from optional arguments
    marker = kwargs.get("marker", dict())
    opacity = marker.get('opacity', 1)
    if opacity < 1:
        log.warning("Opacity < 1 is buggy in Plotly 3D Scatter plot")

    marker['opacity'] = opacity
    marker['size'] = marker.get('size', 1)
    return marker


def get_point_cloud_g_o(bucket, shift=(0, 0, 0), **kwargs):

    if bucket.shape[1] != 3:
        raise ValueError(
            "The shape array is not correct: expected (N,3), got{}".format(bucket.shape))

    bucket = bucket+shift
    x, y, z = bucket.T

    marker = _get_marker_kwarg_for_scatterplot(**kwargs)

    s3d = go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=marker,
        name=kwargs.get('name', None),
    )

    return s3d


# def draw_numpy_bucket(bucket, *, fig=None, **kwargs):
#     """Draw one bucket from numpy array

#     Args:
#         bucket (numpy.ndarray, shape (N,3)): bucket

#     Returns:
#         plotly.graphic_objects.Figure: a plotly figure representing the bucket
#     """

#     assert bucket.shape[1] == 3,\
#         "wrong shape: expected (N,3) got {}".format(bucket.shape)

#     # if shift is not None:
#     #     shift = numpy.array(shift).reshape(1,3)
#     # else:
#     #     shift = numpy.zeros(3)

#     x, y, z = (bucket).T
#     print(fig)
#     if fig is None:
#         fig = go.Figure()

#     marker = _get_marker_kwarg_for_scatterplot(**kwargs)

#     fig.add_trace(
#         go.Scatter3d(x=x, y=y, z=z,
#                      mode='markers',
#                      marker=marker,
#                      name=kwargs.get('name', None))
#     )
#     return fig


# def draw_numpy_buckets(buckets, labels=None,
#                        transpose=True, shift=(0, 0, 0), fig=None):
#     """Draw buckets from numpy arrays.

#     :param buckets: a list of arrays representing the buckets to plot.
#         Each array must be a Nx3 numpy.ndarray containing the coordinates of N points
#     :type buckets: Sequence[numpy.ndarray]
#     :param transpose: for plot reasons, transpose each bucket array if True, defaults to True
#     :type transpose: bool, optional
#     :param x_shift: Amount of x shift to add to each sulcus, defaults to 0
#     :type x_shift: num, optional

#     """

#     gos = []

#     if not isinstance(buckets, dict):
#         if labels is None:
#             labels = range(len(buckets))
#         buckets = dict(zip(labels, buckets))

#     if fig is None:
#         fig = go.Figure()

#     shift = numpy.array(shift)

#     for i, (name, sulcus) in enumerate(buckets.items()):
#         sulcus = sulcus+shift*i

#         if transpose:
#             sulcus = sulcus.T

#         assert sulcus.shape[0] == 3,\
#             "The shape array is not correct, first dimension should be 3, got{}".format(
#                 sulcus.shape[0])

#         x, y, z = sulcus

#         try:
#             len(x)
#         except:
#             # x is a number and not iterable
#             raise ValueError(
#                 "Only one point found in the sulcus, make sure you are providing a list of sulci to the plot function.")

#         # do not set opacity<1 because of a plotpy issue:
#         # https://github.com/plotly/plotly.js/issues/2717

#         gos.append(
#             go.Scatter3d(x=x, y=y, z=z, mode='markers',
#                          marker=dict(size=1, opacity=1),
#                          name=name)
#         )

#     for g in gos:
#         fig.add_trace(g)

#     # markers in the legend are of fixed (big) size
#     fig.update_layout(legend={'itemsizing': 'constant'})

#     return fig