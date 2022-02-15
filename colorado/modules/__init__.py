from . import aims_drawing
from . import numpy_drawing
from . import dico_toolbox_drawing


drawing_functions = dict()

# numpy drawing functions are loaded by default
drawing_functions.update(numpy_drawing.drawing_functions)

# aims drawing functions
if aims_drawing.has_aims:
    drawing_functions.update(aims_drawing.drawing_functions)

# dico_toolbox drawing functions
if dico_toolbox_drawing.has_dtb:
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

    if (f is None) and aims_drawing.has_aims:
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
    if (not is_drawable) and aims_drawing.has_aims:
        is_drawable = aims_drawing.is_aims_volume(obj)

    return is_drawable
