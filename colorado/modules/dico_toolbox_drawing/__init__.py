from ...graphic_objects import *
from ...config import has_brainvisa_and_dico_toolbox as _hbv

if _hbv:
    import dico_toolbox as _dtb

if _hbv:
    drawing_functions = {
        _dtb.wrappers.PyMesh: draw_pyMesh,
        _dtb.wrappers.PyMeshFrame: draw_pyMesh,
    }
else:
    drawing_functions = {}
