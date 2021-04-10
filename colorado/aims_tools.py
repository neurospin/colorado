from soma import aims
import numpy as np


def ndarray_to_volume_aims(ndarray):
    """Create a new volume from the numpy ndarray.
    The array is first converted to Fortran ordering,
    as requested by the Volume constructor

    :param ndarray: the volume data
    :type ndarray: numpy.ndarray
    :return: an Aims Volume object containing the same data as ndarray
    :rtype: aims.Volume
    """
    return aims.Volume(np.asfortranarray(ndarray))


def new_volume_aims_like(vol):
    """Create a new empty aims.Volume with the same shape as vol

    :param vol: the volume data
    :type vol: np.ndarray
    :return: an empty Aims Volume object of the same shape as vol
    :rtype: aims.Volume
    """
    # set same dimensions and data type
    new_vol = aims.Volume(
        vol.getSizeX(),
        vol.getSizeY(),
        vol.getSizeZ(),
        vol.getSizeT(),
        dtype=vol.arraydata().dtype
    )
    # copy the header
    new_vol.header().update(vol.header())
    return new_vol


def bucket_aims_to_ndarray(aims_bucket):
    """Transform an aims bucket into numpy array

    :param aims_bucket: aims bucket object
    :type aims_bucket: soma.aims.BucketMap_VOID.Bucket
    :return: a Nx3 array of points of the bucket
    :rtype: numpy.ndarray
    """
    assert isinstance(aims_bucket, aims.BucketMap_VOID.Bucket)

    v = np.empty((aims_bucket.size(), len(aims_bucket.keys()[0].list())))
    for i, point in enumerate(aims_bucket.keys()):
        v[i] = point.arraydata()

    return v


def volume_to_ndarray(volume):
    """Transform aims volume in numpy array

    Args:
        volume (aims.volume): aims volume
    """
    return volume[:].squeeze()

def ndarray_to_aims_volume(ndarray):
    """Create a new volume with the data in ndarray.

    The array is first converted to Fortran ordering,
    as requested by the Volume constructor."""
    return aims.Volume(np.asfortranarray(ndarray))


def bucket_numpy_to_volume_numpy(bucket_array, pad=0):
    """Transform a bucket into a 3d boolean volume.
    Input and output types are numpy.ndarray"""
    a = bucket_array
    v_max = a.max(axis=0)
    v_min = a.min(axis=0)
    v_size = abs(v_max - v_min) + 1 + pad*2

    vol = np.zeros(v_size)

    for p in a:
        x, y, z = np.round(p-v_min+pad).astype(int)
        vol[x, y, z] = 1

    return vol

def bucket_numpy_to_volume_aims(bucket_array, pad=0):
    """Transform a bucket into a 3d boolean volume.
    Input and output types are numpy.ndarray"""
    a = bucket_array
    v_max = a.max(axis=0)
    v_min = a.min(axis=0)
    v_size = abs(v_max - v_min) + 1 + pad*2

    vol = aims.Volume(*v_size, dtype='int16')
    vol.fill(0)
    avol = vol[:].squeeze()

    for p in a:
        x, y, z = np.round(p-v_min+pad).astype(int)
        avol[x, y, z] = 1

    return vol


def bucket_aims_to_volume_aims(aims_bucket, pad=0):
    """Transform a bucket into a 3d boolean volume.
    Input and output types are aims objects"""
    abucket = bucket_aims_to_ndarray(aims_bucket)
    return bucket_numpy_to_volume_aims(abucket, pad=pad)


def volume_to_bucket_numpy(volume):
    """Transform a binary volume into a bucket.
    The bucket contains the coordinates of the non-zero voxels in volume.

    Args:
        volume (numpy array | aims volume): 3D image

    Returns:
        numpy.ndarray: bucket of non-zero points coordinates
    """
    return np.argwhere(volume[:].squeeze())

def volume_to_bucket_aims(volume):
    return  np.argwhere(volume[:].squeeze())

def add_border(x, thickness, value):
    """add borders to volume (numpy)"""
    t = thickness
    x[:t, :, :] = value
    x[-t:, :, :] = value

    x[:, :t, :] = value
    x[:, -t:, :] = value

    x[:, :, :t] = value
    x[:, :, -t:] = value

    return x


class PyMesh:
    def __init__(self, aims_mesh=None):
        """A multi-frame mesh.

        Args:
            aims_mesh ([aims mesh], optional): if an aims mesh is passed to the class constructor,
            the mesh data is copied into the new object. Defaults to None.

        Raises:
            ValueError: when the aims_mesh is not conform.
        """
        self.frames = [PyMeshFrame()]
        self.header = {}
        if aims_mesh is not None:
            self.header = aims_mesh.header()
            self.frames = [None]*aims_mesh.size()
            for i in range(aims_mesh.size()):
                l = PyMeshFrame()
                try:
                    l.vertices = PyMesh._mesh_prop_to_numpy(
                        aims_mesh.vertex(i))
                    l.polygons = PyMesh._mesh_prop_to_numpy(
                        aims_mesh.polygon(i))
                    l.normals = PyMesh._mesh_prop_to_numpy(aims_mesh.normal(i))
                except:
                    raise ValueError("Invalid aims mesh")
                self.frames[i] = l

    @property
    def vertices(self):
        return self.frames[0].vertices

    @vertices.setter
    def vertices(self, v):
        self.frames[0].vertices = v

    @property
    def polygons(self):
        return self.frames[0].polygons

    @polygons.setter
    def polygon(self, v):
        self.frames[0].polygons = v

    @property
    def normals(self):
        return self.frames[0].normals

    @normals.setter
    def normals(self, v):
        self.frames[0].normals = v

    def append(self, frame):
        self.frames.append(frame)

    def __getitem__(self, i):
        return self.frames[i]

    def __setitem__(self, i, v):
        self.frames[i] = v

    def __len__(self):
        return len(self.frames)

    def __repr__(self):
        return "Mesh of {} frame(s)\n{}".format(
            len(self.frames),
            '\n'.join([str(k)+': '+str(v) for k, v in self.header.items()])
        )

    def to_aims_mesh(self, header={}):
        """Get the aims mesh version of this mesh."""

        mesh = aims.AimsTimeSurface()

        for i, frame in enumerate(self.frames):
            vertices = frame.vertices.tolist()
            polygons = frame.polygons.tolist()
            mesh.vertex(i).assign(vertices)
            mesh.polygon(i).assign(polygons)

        # update normals
        mesh.updateNormals()
        # update header
        mesh.header().update(header)

        return mesh

    @staticmethod
    def _mesh_prop_to_numpy(mesh_prop):
        """return a new numpy array converting AIMS mesh properties
        into numpy ndarrays (soma.aims.vector_POINT2DF)"""
        return np.array([x[:] for x in mesh_prop])


class PyMeshFrame:
    def __init__(self, frame=None):
        """One frame of a mesh with numpy array vertices, polygons and normals."""
        self.vertices = None
        self.polygons = None
        self.normals = None
        self.header = None

        if frame is not None:
            self.vertices = frame.vertices.copy()
            self.polygons = frame.polygons.copy()
            self.normals = frame.normals.copy()
            self.header = frame.header

    def __repr__(self):
        if self.polygons is not None:
            ln = len(self.polygons)
        else:
            ln = 0
        return "PyMeshFrame of {} triangles\n".format(ln)
