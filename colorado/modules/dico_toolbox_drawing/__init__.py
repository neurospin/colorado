from ...graphic_objects import *
import logging
log = logging.getLogger(__name__)

has_dtb = True

try:
    import dico_toolbox
except ImportError:
    # module is not available
    has_dtb = False


if has_dtb:
    drawing_functions = {
        dico_toolbox.wrappers.PyMesh: draw_pyMesh,
        dico_toolbox.wrappers.PyMeshFrame: draw_pyMesh,
    }
else:
    drawing_functions = {}
