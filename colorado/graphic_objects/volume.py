from numpy.core import numeric
from .point_cloud import get_point_cloud_g_o
import numpy as np
import plotly.graph_objects as go
from ..config import has_brainvisa_and_dico_toolbox as _hbv

if _hbv:
    from dico_toolbox import _aims_tools


def get_volume_g_o(volume,
                   max_points: int = 10000,
                   downsample: int = None,
                   th_min: numeric = None, th_max: numeric = None,
                   shift=(0, 0, 0),
                   **kwargs):
    """Get a volume as a scatter plot.

    Threshold is applied if the th_min and th_max values are defined.

    Args:
        volume (ndarray): the volume to plot
        max_points (int, optional): Max number of points to plot. Defaults to 10000.
        downsample (int, optionale): donwsample the volume by this factor, defaults to None (no downsample)
            One-every-downsample voxels are removed from the plot. The volume is not rescaled
        th_min (num, optional): lower threshold value. Defaults to None
        th_max (num, optional): higher threshold value. Defaults to None.  

    Returns:
        plotly.grapthic_object: a gtaphic object to be added to a plotly figure
    """
    avol = volume.copy()

    # apply threshold
    if th_min is not None:
        avol[avol < th_min] = 0
    if th_max is not None:
        avol[avol > th_max] = 0

    # downsample the volume
    if downsample is not None:
        temp = np.zeros_like(avol)
        temp[::downsample, ::downsample,
             ::downsample] = avol[::downsample, ::downsample, ::downsample]
        avol = temp

    apc = np.argwhere(avol)

    # limit number of points
    if len(apc) > max_points:
        idx = np.random.randint(0, len(apc), size=max_points)
        apc = apc[idx]

    # get the values
    if kwargs.get('use_values', None):
        x, y, z = apc.T
        values = avol[x, y, z]
        go = get_point_cloud_g_o(apc, values=values, shift=shift, **kwargs)
    else:
        go = get_point_cloud_g_o(apc, shift=shift, **kwargs)

    return go


def draw_volume(volume, fig=None,
                max_points=10000,
                downsample=None,
                th_min=None, th_max=None,
                **kwargs):
    """Draw a volume"""
    if fig is None:
        fig = go.Figure()
    g = get_volume_g_o(
        volume, max_points=max_points,
        downlsample=downsample,
        th_min=th_min, th_max=th_max,
        **kwargs
    )
    fig.add_trace(g)
    return fig


# def draw_volumes(volumes, fig=None,
#                  max_points=10000,
#                  downsample=None,
#                  th_min=None, th_max=None,
#                  labels=None, shift=(0, 0, 0),
#                  **kwargs):
#     """Draw volumes"""
#     if fig is None:
#         fig = go.Figure()

#     if not isinstance(volumes, dict):
#         labels = range(len(volumes))
#         volumes = dict(zip(labels, volumes))

#     shift = np.array(shift)

#     for i, (name, volume) in enumerate(volumes.items()):
#         g = get_volume_g_o(
#             volume, max_points=max_points,
#             downlsample=downsample,
#             th_min=th_min, th_max=th_max,
#             name=name, shift=shift*i,
#             **kwargs
#         )
#         fig.add_trace(g)

#     fig.update_layout(legend={'itemsizing': 'constant'})
#     return fig
