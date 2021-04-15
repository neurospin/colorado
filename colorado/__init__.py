import plotly
import numpy
from .anatomist_tools import anatomist_snatpshot
from .bucket import get_aims_bucket_g_o, draw_numpy_bucket, draw_numpy_buckets, get_aims_bucket_map_g_o
from .mesh import get_aims_mesh_g_o, draw_pyMesh, draw_meshes_in_subplots, draw_numpy_meshes
from .volume import draw_volume, get_volume_g_o, draw_volumes
from re import match as _re_match

from .aims_tools import PyMesh, PyMeshFrame
from . import aims_tools

import numpy

from soma import aims as _aims


def draw(data, fig=None, labels=None, shift=(0, 0, 0), draw_function=None, draw_f_args=dict(), **kwargs):
    """Draw objects with plotly

    :param data: an object or a list of objects. Object can be an aims bucket, an aims Mesh or a PyMesh
    :type data: soma.aims mesh/volume/bucket 
    :param fig: the plotly figure, if None, one is created, defaults to None
    :type fig: plotly.graph_objects.Figure, optional
    :param labels: an array of labels for the data if data is a list, defaults to None
    :type labels: list[str], optional
    :param shift: shift the plots in data each by (x,y,z), defaults to (0,0,0)
    :type shift: tuple, optional
    :return: a plotly figure
    :rtype: plotly.graph_objects.Figure
    """

    # check if the object is iterable
    if isinstance(data, dict):
        # if it is a dictionary, use keys as labels
        labels = list(data.keys())
        data = list(data.values())
    elif not isinstance(data, list):
        data = [data]

    if fig is None:
        fig = plotly.graph_objects.Figure()

    if labels is None:
        labels = [None]*len(data)

    if isinstance(data, list):
        assert len(data) == len(labels)

    shift = numpy.array(shift)

    for i, obj in enumerate(data):
        if draw_function is None:
            f = _get_draw_function(obj)
        else:
            f = draw_function
        trace = f(obj, name=labels[i], shift=shift*i, **draw_f_args, **kwargs)

        if isinstance(trace, plotly.basedatatypes.BaseTraceHierarchyType):
            fig.add_trace(trace)
        else:
            # trace might be a list of traces
            try:
                trace = iter(trace)
                for tr in trace:
                   fig.add_trace(tr) 
            except:
                # raise
                raise ValueError("Dwawind Error")

    fig.update_layout(legend={'itemsizing': 'constant'})

    return fig


def _raise_numpy_error(obj, name, shift, **kwargs):
    raise ValueError(
        "numpy object are ambiguous and can not be drawn. Use a specific function (e.g. colorado.draw_volume)")


_drawing_functions = {
    _aims.AimsTimeSurface_3_VOID: get_aims_mesh_g_o,
    _aims.BucketMap_VOID.Bucket: get_aims_bucket_g_o,
    PyMesh: draw_pyMesh,
    PyMeshFrame: draw_pyMesh,
    _aims.Volume_S16: get_volume_g_o,
    numpy.ndarray: _raise_numpy_error,
    _aims.BucketMap_VOID: get_aims_bucket_map_g_o
}


def _get_draw_function(obj):
    """Get the appropriate drawing function for obj

    Args:
        obj (object): an object

    Raises:
        ValueError: the object is not drawable

    Returns:
        function: the drawing function
    """
    f = _drawing_functions.get(type(obj), None)

    if f is None:
        is_volume = _re_match(r".*soma.aims.Volume.*", str(type(obj)))
        is_aims_data = _re_match(r".*soma.aims.AimsData.*", str(type(obj)))
        if is_volume or is_aims_data:
            f = _drawing_functions[_aims.Volume_S16]
        else:
            raise ValueError("I don't know how to draw {}".format(type(obj)))

    return f
