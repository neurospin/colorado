from typing import Tuple
import plotly
import plotly.graph_objects as go
import numpy as _np
from .modules import get_draw_function


import logging
log = logging.getLogger(__name__)


def new_figure(*args, **kwargs):
    """Create a new Figure"""
    return plotly.graph_objs.Figure(*args, **kwargs)


def draw(*args, fig=None, label=None, shift=(0, 0, 0), draw_function=None, draw_f_args=dict(), title="", **kwargs):
    """Draw objects with plotly.

    args can be a drawable object or an iterable/dict of drawable objects.
    Drawable objects are:
        - 3D numpy arrays, interpreted as voulmetric images
        - 2D (N,3) arrays, interpreted as point-clouds
        - Brainvisa's Aims Volumes, Buckets and Meshes
        - custom objects that implement a __draw_with_colorado__ method returning either
          a drawable object or a plotly.graph_object object


    Args:
        *args : drawable objects or an iterable/dict of drawable objects.
        fig (plotly.graph_object, optional): A Figure to which this object will be added. Defaults to None.
        label (str, optional): Label of this drawing. Defaults to None.
        shift (tuple, optional): constant offset added to all points of this drawing. Defaults to (0, 0, 0).
        draw_function (function, optional): A function that returns a plotly graph_object. Defaults to None.
        draw_f_args (dict, optional): arguments directly passed to draw_function. Defaults to dict().
        title (str, optional): Title of the Figure. Defaults to "".

    Raises:
        ValueError: The object(s) can not be drawn.

    Returns:
        plotly.graph_objs.Figure : a figure containing the drawing
    """

    # create a labeled dict of objects to draw
    data = _data_to_dict(args, label)

    if fig is None:
        fig = plotly.graph_objects.Figure()

    shift = _np.array(shift)

    for i, (name, obj) in enumerate(data.items()):

        # pass the plot name (label) to the drawing function as kwargs
        kwargs["name"] = name

        try:
            # The object has a special method called by colorado
            trace = obj.__draw_with_colorado__(**draw_f_args, **kwargs)
            if not trace.__class__.__module__.startswith('plotly.graph_objs'):
                # The object has a __draw_with_colorado__ method but the return value
                # is not a plotly graph_object. In this case the drawing function
                # has to be guessed
                raise AttributeError
        except AttributeError:
            # ...otherwise
            if draw_function is None:
                f = get_draw_function(obj)
            else:
                f = draw_function

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

    fig.update_layout(
        scene_aspectmode='data',
        scene_camera=dict(eye=dict(x=-0.8, y=0, z=2)),
        legend={'itemsizing': 'constant'},
        title=title
    )

    # fig.update_scenes(aspectratio=dict(x=1, y=5, z=1))

    return fig


def _data_to_dict(data, label):
    """Create a {label:graphicObject} dictionary"""
    assert(type(data) == tuple)
    # NOTE: aims Volumes is an iterable!
    if len(data) == 1:
        if isinstance(data[0], dict):
            # data is one dictionary
            return data[0]
        elif type(data[0]) in (tuple, list):
            # data is one list/tuple
            if label is None:
                label = ["trace {}".format(x) for x in range(len(data[0]))]
            data = dict(zip(label, data[0]))
        else:
            # data is one object (not list/tuple/dict)
            data = {label: data[0]}
    else:
        # the user gave more than one argument
        assert(all([type(d) not in [list, tuple, dict] for d in data]))
        if label is None:
            label = ["trace {}".format(x) for x in range(len(data))]
        assert len(data) == len(label)
        data = dict(zip(label, data))

    return data


def draw_as_mesh(*args):
    raise DeprecationWarning(
        "This function is deprecated. Use dico_toolbox to convert into mesh and then draw().")
