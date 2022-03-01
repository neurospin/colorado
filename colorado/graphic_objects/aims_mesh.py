from numpy.lib.polynomial import poly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .mesh import get_mesh3d_g_o
from ..config import has_brainvisa_and_dico_toolbox as _hbv

if _hbv :
    from soma import aims as _aims
    from dico_toolbox.wrappers import PyMesh


def get_aims_mesh_g_o(mesh, shift=(0, 0, 0), **kwargs):
    """Get a plotly graphic object from an aims mesh"""
    if mesh.__class__.__name__ != "PyMeshFrame":
        if isinstance(mesh, PyMesh):
            mesh = mesh[0]
        else:
            # it was an aims mesh
            mesh = PyMesh(mesh)[0]

        vertices = mesh.vertices
        polygons = mesh.polygons

    return get_mesh3d_g_o(vertices, polygons, shift=shift, **kwargs)


def draw_pyMesh(mesh, shift=(0, 0, 0), **kwargs):
    """Draw a PyMesh object

    Args:
        mesh (PyMesh): The mesh to be drown

    Returns:
        plotly.graphic_object: graphic object
    """
    return get_mesh3d_g_o(mesh.vertices, mesh.polygons,  shift=shift, **kwargs)