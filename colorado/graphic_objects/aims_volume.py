from numpy.core import numeric
from .volume import get_volume_g_o
from ..config import has_brainvisa_and_dico_toolbox as _hbv

if _hbv:
    from dico_toolbox import _aims_tools


def get_aims_volume_g_o(aims_volume,
                   max_points: int = 10000,
                   downsample: int = None,
                   th_min: numeric = None, th_max: numeric = None,
                   shift=(0, 0, 0),
                   **kwargs):
    """Get a volume as a scatter plot.

    Threshold is applied if the th_min and th_max values are defined.

    Args:
        aims_volume (aims.Volume): the volume to plot
        max_points (int, optional): Max number of points to plot. Defaults to 10000.
        downsample (int, optionale): donwsample the volume by this factor, defaults to None (no downsample)
            One-every-downsample voxels are removed from the plot. The volume is not rescaled
        th_min (num, optional): lower threshold value. Defaults to None
        th_max (num, optional): higher threshold value. Defaults to None.  

    Returns:
        plotly.grapthic_object: a gtaphic object to be added to a plotly figure
    """
    avol = _aims_tools.volume_to_ndarray(aims_volume).copy()

    go = get_volume_g_o(avol, max_points, downsample, th_min, th_max, **kwargs)

    return go