import plotly
import numpy
from .anatomist_tools import anatomist_snatpshot
from .bucket import get_aims_bucket_g_o, draw_numpy_bucket ,draw_numpy_buckets
from .mesh import get_aims_mesh_g_o, draw_pyMesh, draw_meshes_in_subplots, draw_numpy_meshes
from .volume import draw_binary_volume

from .aims_tools import PyMesh, PyMeshFrame

import numpy

from soma import aims as _aims


def draw(data, fig=None, labels=None, shift=(0, 0, 0), **kwargs):
    """Draw objects with plotly

    :param data: an object or a list of objects. Object can be an aims bucket, an aims Mesh or a PyMesh
    :type data: soma.aims.AimsTimeSurface_3_VOID | soma.aims.BucketMap_VOID.Bucket 
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
    if not isinstance(data, list):
        data = [data]

    drawing_f = {
        _aims.AimsTimeSurface_3_VOID: get_aims_mesh_g_o,
        _aims.BucketMap_VOID.Bucket: get_aims_bucket_g_o,
        PyMesh: draw_pyMesh,
        PyMeshFrame: draw_pyMesh,
        _aims.Volume_S16: draw_binary_volume
    }

    if fig is None:
        fig = plotly.graph_objects.Figure()

    if labels is None:
        labels = [None]*len(data)

    if isinstance(data, list):
        assert len(data) == len(labels)

    shift = numpy.array(shift)

    for i, obj in enumerate(data):
        try:
            go = drawing_f[type(obj)](obj, name=labels[i], shift=shift*i, **kwargs)
            fig.add_trace(go)
        except KeyError as e:
            raise ValueError("I don't know how to draw {}".format(type(obj)))

    fig.update_layout(legend={'itemsizing': 'constant'})

    return fig