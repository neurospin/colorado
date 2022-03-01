from colorado.graphic_objects.mesh import get_mesh3d_g_o


class SimpleMesh:
    def __init__(self, vertices=None, polygons=None):
        """A simple Mesh object, drawable with colorado"""
        self.header = {}
        self._vertices = vertices
        self._polygons = polygons
        self._normals = None

    @ property
    def vertices(self):
        return self._vertices

    @ property
    def polygons(self):
        return self._polygons
    
    @ property
    def normals(self):
        return self._normals

    def set_vertices(self, v):
        self._vertices = v

    def set_polygons(self, v):
        self._polygons = v

    def set_normals(self, v):
        self._normals = v

    def __repr__(self):
        return "Mesh of {} polygons".format(len(self._polygons))
    
    def to_dict(self):
        return {
            "vertices": self.vertices,
            "polygons": self.polygons,
            "normals": self.normals
        }
        
    def __draw_with_colorado__(self, shift=(0,0,0), **kwargs):
        return get_mesh3d_g_o(self.vertices, self.polygons, shift=shift, **kwargs)