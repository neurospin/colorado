from numpy.lib.arraysetops import isin
import plotly
import numpy
from .anatomist_tools import anatomist_snatpshot
from .bucket import get_aims_bucket_g_o, draw_numpy_bucket, draw_numpy_buckets, get_aims_bucket_map_g_o
from .mesh import get_aims_mesh_g_o, draw_pyMesh, draw_meshes_in_subplots, draw_numpy_meshes
from .volume import draw_volume, get_volume_g_o, draw_volumes
from re import match as _re_match

from .aims_tools import PyMesh, PyMeshFrame, buket_to_aligned_mesh, buket_to_mesh, volume_to_mesh
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

    # create a labeled dict of objects to draw
    data = _data_to_dict(data, labels)

    if fig is None:
        fig = plotly.graph_objects.Figure()

    shift = numpy.array(shift)

    for i, (name, obj) in enumerate(data.items()):

        if draw_function is None:
            f = _get_draw_function(obj)
        else:
            f = draw_function
        trace = f(obj, name=name, shift=shift*i, **draw_f_args, **kwargs)

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
                raise ValueError("Drawing Error")

    fig.update_layout(legend={'itemsizing': 'constant'})

    return fig


def _data_to_dict(data, labels):
    # check if the object is iterable
    if not isinstance(data, dict):
        # Check for instance of list instead iterability because
        # aims Volumes is an iterable
        if not isinstance(data, list):
            data = [data]
        if labels is None:
            labels = ["trace {}".format(x) for x in range(len(data))]
        assert len(data) == len(labels)
        data = dict(zip(labels, data))

    return data


def draw_as_mesh(data, gaussian_blur_FWWM=0, threshold_quantile=0, labels=None, shift=(0, 0, 0), **kwargs):

    # create a labeled dict of objects to draw
    data = _data_to_dict(data, labels)

    for name, obj in data.items():
        if isinstance(data, numpy.ndarray):
            raise ValueError(
                "numpy object are ambiguous. use colorado.aims_tools functions to convert them into aims objects")
        elif _is_aims_volume(obj):
            data[name] = aims_tools.volume_to_mesh(
                obj, gaussian_blur_FWWM=gaussian_blur_FWWM, threshold_quantile=threshold_quantile)
        elif isinstance(obj, _aims.BucketMap_VOID.Bucket):
            data[name] = aims_tools.bucket_to_mesh(
                obj, gaussian_blur_FWWM=gaussian_blur_FWWM, threshold_quantile=threshold_quantile)
        elif isinstance(obj, _aims.AimsTimeSurface_3_VOID) or isinstance(obj, PyMesh):
            # it's already a mesh
            pass
        else:
            # it's something else
            raise ValueError(
                "Oups, I can't convert a {} object into aims mesh".format(type(obj)))

    return draw(data, shift=shift, **kwargs)


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


def _is_aims_volume(obj):
    return _re_match(r".*soma.aims.Volume.*", str(type(obj))) or _re_match(r".*soma.aims.AimsData.*", str(type(obj)))


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
        if _is_aims_volume(obj):
            f = _drawing_functions[_aims.Volume_S16]
        else:
            raise ValueError("I don't know how to draw {}".format(type(obj)))

    return f
