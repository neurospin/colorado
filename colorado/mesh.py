from numpy.lib.polynomial import poly
from .aims_tools import PyMesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from soma import aims as _aims


def get_mesh3d_g_o(vertices, polygons, name=None, shift=(0, 0, 0), **kwargs):
    """Get an plotly graphic object from numpy arrays

    Args:
        vertices (numpy.ndarray): Nx3 array of points representing the vertices
        polygons (numpy.ndarray): Nx3 array of points representing the polygons
        name (str, optional): The plot name. Defaults to None.

    Returns:
        plotly.graph_objects : a graphic pbject
    """
    v = vertices + shift
    v = v.T
    p = polygons.T
    return go.Mesh3d(
        x=v[0], y=v[1], z=v[2],
        i=p[0], j=p[1], k=p[2],
        opacity=1,
        color='rgb(255,200,200)',
        lighting=go.mesh3d.Lighting(ambient=0.1,),
        lightposition=go.mesh3d.Lightposition(x=0, y=0, z=0),
        name=name
    )


def get_aims_mesh_g_o(mesh, name=None, shift=(0, 0, 0), **kwargs):
    """Get a plotly graphic object from an aims mesh"""
    if mesh.__class__.__name__ != "PyMeshFrame":
        if isinstance(mesh, PyMesh):
            mesh = mesh[0]
        else:
            # it was an aims mesh
            mesh = PyMesh(mesh)[0]

        vertices = mesh.vertices
        polygons = mesh.polygons

    return get_mesh3d_g_o(vertices, polygons, name=name, shift=shift, **kwargs)


def draw_meshes_in_subplots(mesh_list, cols=3):
    """Draw meshes in a subplot grid.

    Args:
        mesh_list (list[aims mesh | PyMesh]): a list of meshes to be drawn
        cols (int, optional): Number of columns in the subplot. Defaults to 3.

    Returns:
        plotly.graphic_objects.Figure: the complete figure
    """
    cols = min(cols, len(mesh_list))
    rows = int((len(mesh_list)//cols))

    if len(mesh_list) % cols != 0:
        rows += 1

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'is_3d': True}]*cols]*rows
    )

    for i, mesh in enumerate(mesh_list):

        r = i//cols+1
        c = i % cols+1

        if isinstance(mesh, _aims.AimsTimeSurface_3_VOID):
            go = get_aims_mesh_g_o(mesh)
        else:
            go = get_mesh3d_g_o(mesh.vertices, mesh.polygons)
        fig.append_trace(go, row=r, col=c)

    return fig


def draw_pyMesh(mesh, shift=(0, 0, 0), **kwargs):
    """Draw a PyMesh object

    Args:
        mesh (PyMesh): The mesh to be drown

    Returns:
        plotly.graphic_object: graphic object
    """
    return get_mesh3d_g_o(mesh.vertices, mesh.polygons,  shift=shift, **kwargs)


def draw_numpy_meshes(list_of_vertices, list_of_polygons, labels=None):
    """Draw meshes from the vertices and polygons of the mesh.
    The meshes are drown in the same figure.

    Args:
        list_of_vertices (list[numpy.array]): lists of array of points.
        list_of_polygons (list[numpy.array]): lists of array of points.
        labels (list[str], optional): list of plots labels. Defaults to None.

    Returns:
        plotly.graphic_objects.Figure: the complete figure
    """
    fig = go.Figure()

    if labels is None:
        labels = [None]*len(list_of_polygons)

    assert len(list_of_polygons) == len(list_of_vertices)
    assert len(list_of_polygons) == len(labels)

    for i in range(len(list_of_polygons)):
        vertices = list_of_vertices[i]
        polygons = list_of_polygons[i]
        name = labels[i]
        fig.add_trace(get_mesh3d_g_o(vertices, polygons))

    return fig
