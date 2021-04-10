from numpy.lib.shape_base import _make_along_axis_idx
from . import aims_tools
from .bucket import get_bucket_g_o
import numpy as np
import plotly.graph_objects as go



def get_volume_g_o(volume,
                   max_points=10000,
                   th_min=None, th_max=None,
                   **kwargs):
    """Get a volume as a scatter plot.

    Threshold is applied if the th_min and th_max values are defined.

    Args:
        aims_volume (aims.Volume): the volume to plot
        max_points (int, optional): Max number of points to plot. Defaults to 10000.
        th_min (num, optional): lower threshold value. Defaults to None
        th_max (num, optional): higher threshold value. Defaults to None.  

    Returns:
        plotly.grapthic_object: a gtaphic object to be added to a plotly figure
    """

    avol = aims_tools.volume_to_ndarray(volume)

    # apply threshold
    if th_min is not None:
        avol[avol < th_min] = 0
    if th_max is not None:
        avol[avol > th_max] = 0

    abucket = aims_tools.volume_to_bucket_numpy(avol)

    # limit number of points
    if len(abucket) > max_points:
        idx = np.random.randint(0, len(abucket), size=max_points)
        abucket = abucket[idx]

    # get the values
    if kwargs.get('use_values', None):
        x, y, z = abucket.T
        values = avol[x, y, z]
        go = get_bucket_g_o(abucket, values=values, **kwargs)
    else:
        go = get_bucket_g_o(abucket, **kwargs)

    return go


def draw_volume(volume, fig=None,
                max_points=10000,
                th_min=None, th_max=None,
                **kwargs):
    """Draw a volume"""
    if fig is None:
        fig = go.Figure()
    g = get_volume_g_o(
        volume, max_points=max_points,
        th_min=th_min, th_max=th_max,
        **kwargs
    )
    fig.add_trace(g)
    return fig
