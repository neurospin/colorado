from numpy.lib.arraysetops import isin
import plotly
import numpy as _np
from .anatomist_tools import anatomist_snatpshot
from .bucket import get_aims_bucket_g_o, draw_numpy_bucket, draw_numpy_buckets, get_aims_bucket_map_g_o
from .mesh import get_aims_mesh_g_o, draw_pyMesh, draw_meshes_in_subplots, draw_numpy_meshes
from .volume import draw_volume, get_volume_g_o, draw_volumes, get_bucket_g_o
from re import match as _re_match

import dico_toolbox as _dt
from soma import aims as _aims

import logging
log = logging.getLogger(__name__)

def new_figure(*args, **kwargs):
    """Create a new Figure"""
    return plotly.graph_objs.Figure(*args, **kwargs)


def draw(*args, fig=None, label=None, shift=(0, 0, 0), draw_function=None, draw_f_args=dict(), **kwargs):
    """Draw objects with plotly

    :param data: drawable objects or an iterable of drawable objects.
    :type data: soma.aims mesh/volume/bucket or numpy array
    :param fig: the plotly figure, if None, one is created, defaults to None
    :type fig: plotly.graph_objects.Figure, optional
    :param label: a string or an iterable of labels for the data.
    :type label: list[str], optional
    :param shift: shift the plots in data each by (x,y,z), defaults to (0,0,0)
    :type shift: tuple, optional
    :return: a plotly figure
    :rtype: plotly.graph_objects.Figure
    """

    # create a labeled dict of objects to draw
    data = _data_to_dict(args, label)

    if fig is None:
        fig = plotly.graph_objects.Figure()

    shift = _np.array(shift)

    for i, (name, obj) in enumerate(data.items()):

        if draw_function is None:
            f = _get_draw_function(obj)
        else:
            f = draw_function
        
        kwargs["name"] = name

        trace = f(obj, shift=shift*i, **draw_f_args, **kwargs)

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
    fig.update_layout(scene_aspectmode='data')

    return fig


def _is_drawable(obj):
    return type(obj) in _drawing_functions.keys()

def _data_to_dict(data, label):
    """Create a {label:graphicObject} dictionary"""
    assert(type(data) == tuple)
    # NOTE: aims Volumes is an iterable!
    if len(data) == 1:
        if isinstance(data[0], dict):
            # data is one dictionary
            pass
        elif type(data[0]) in (tuple, list):
            # data is one list/tuple
            if label is None:
                label = ["trace {}".format(x) for x in range(len(data[0]))]
            data = dict(zip(label, data[0]))
        else:
            # data is one object (not list/tuple/dict)
            data = {label : data[0]}
    else:
        # the user gave more than one argument
        assert( all([type(d) not in [list, tuple, dict] for d in data]))
        if label is None:
            label = ["trace {}".format(x) for x in range(len(data))]
        assert len(data) == len(label)
        data = dict(zip(label, data))
    
    return data


def draw_as_mesh(data, gaussian_blur_FWWM=0, threshold_quantile=0, labels=None, shift=(0, 0, 0), **kwargs):

    # create a labeled dict of objects to draw
    data = _data_to_dict(data, labels)

    for name, obj in data.items():
        if isinstance(data, numpy.ndarray):
            raise ValueError(
                "numpy object are ambiguous. It may be safer to unse a dedicated function (e.g. draw_numpy_bucket)")
        elif _is_aims_volume(obj):
            data[name] = _dt.convert.volume_to_mesh(obj)
        elif isinstance(obj, _aims.BucketMap_VOID.Bucket):
            data[name] = _dt.convert.bucket_to_mesh(obj)
        elif isinstance(obj, _aims.AimsTimeSurface_3_VOID) or isinstance(obj, _dt.wrappers.PyMesh):
            # it's already a mesh
            pass
        else:
            # it's something else
            raise ValueError(
                "Oups, I can't convert a {} object into aims mesh".format(type(obj)))

    return draw(data, shift=shift, **kwargs)


def _process_numpy_object(obj, **kwargs):
    # raise ValueError(
    #     "numpy object are ambiguous and can not be drawn. Use a specific function (e.g. colorado.draw_volume)")
    log.debug("Numpy object are ambiguous. Use a specific function (e.g. colorado.draw_volume)")
    if len(obj.shape) == 2 and obj.shape[1] == 3:
        f = get_bucket_g_o
    elif len(obj.shape) == 3:
        f = get_volume_g_o
    else:
        raise ValueError(f"Could not interpretate numpy object of shape {obj.shape} as bucket or volume.")

    return f(obj, **kwargs)

_drawing_functions = {
    _aims.AimsTimeSurface_3_VOID: get_aims_mesh_g_o,
    _aims.BucketMap_VOID.Bucket: get_aims_bucket_g_o,
    _dt.wrappers.PyMesh: draw_pyMesh,
    _dt.wrappers.PyMeshFrame: draw_pyMesh,
    _aims.Volume_S16: get_volume_g_o,
    _np.ndarray: _process_numpy_object,
    _aims.BucketMap_VOID: get_aims_bucket_map_g_o,
    _aims.rc_ptr_BucketMap_VOID: get_aims_bucket_map_g_o
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
