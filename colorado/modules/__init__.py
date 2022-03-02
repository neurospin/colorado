from . import aims_drawing
from . import numpy_drawing
from ..config import has_brainvisa_and_dico_toolbox as _hbv
if _hbv:
    # brainvisa and dico_toolbox are available
    from . import dico_toolbox_drawing

import logging as _logging
_log = _logging.getLogger(__name__)

drawing_functions = dict()

# numpy drawing functions are loaded by default
drawing_functions.update(numpy_drawing.drawing_functions)


if _hbv:
    _log.info("updating drawing functions with brainvisa")
    # aims drawing functions
    drawing_functions.update(aims_drawing.drawing_functions)
    # dico_toolbox drawing functions
    drawing_functions.update(dico_toolbox_drawing.drawing_functions)


def get_draw_function(obj):
    """Get the appropriate drawing function for obj

    Args:
        obj (object): an object

    Raises:
        ValueError: the object is not drawable

    Returns:
        function: the drawing function
    """
    f = drawing_functions.get(type(obj), None)

    if (f is None) and _hbv:
        if aims_drawing.is_aims_volume(obj):
            f = drawing_functions[aims_drawing.volume_function]
        else:
            raise ValueError("I don't know how to draw {}".format(type(obj)))

    return f


def is_drawable(obj):
    """Check wether the object is drawable with colorado.draw()"""

    # check for all drawable objects
    is_drawable = type(obj) in drawing_functions.keys()

    # check for aims volume
    if (not is_drawable) and _hbv:
        is_drawable = aims_drawing.is_aims_volume(obj)

    return is_drawable
