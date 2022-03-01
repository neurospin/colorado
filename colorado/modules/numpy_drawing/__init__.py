from ...graphic_objects import *

import numpy
import logging
log = logging.getLogger(__name__)


def process_numpy_object(obj, **kwargs):
    # raise ValueError(
    #     "numpy object are ambiguous and can not be drawn. Use a specific function (e.g. colorado.draw_volume)")
    log.debug(
        "Numpy object are ambiguous. Use a specific function (e.g. colorado.draw_volume)")
    if len(obj.shape) == 2 and obj.shape[1] == 3:
        f = get_point_cloud_g_o
    elif len(obj.shape) == 3:
        f = get_volume_g_o
    else:
        raise ValueError(
            f"Could not interpretate numpy object of shape {obj.shape} as bucket or volume.")

    return f(obj, **kwargs)


drawing_functions = {numpy.ndarray: process_numpy_object}
