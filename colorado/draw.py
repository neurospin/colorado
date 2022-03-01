import plotly
import numpy as _np
from .modules import get_draw_function


import logging
log = logging.getLogger(__name__)


def new_figure(*args, **kwargs):
    """Create a new Figure"""
    return plotly.graph_objs.Figure(*args, **kwargs)


def draw(*args, fig=None, label=None, shift=(0, 0, 0), draw_function=None, draw_f_args=dict(), title="", **kwargs):
    """Draw objects with plotly.

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
        
        # pass the plot name (label) to the drawing function as kwargs
        kwargs["name"] = name
        
        try:
        # The object has a special method called by colorado
            trace = obj.__draw_with_colorado__(**draw_f_args, **kwargs)
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
