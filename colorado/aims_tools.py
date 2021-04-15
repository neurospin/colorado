from soma import aims, aimsalgo
import numpy as np
from scipy.ndimage import gaussian_filter


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
    # take the firts element of the last axis instead of squeeze avoids
    # problems with volumens that have last dimension > 1.
    if not isinstance(volume, np.ndarray):
        return volume[:, :, :, 0]
    else:
        return volume


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
    v_size = np.ceil(abs(v_max - v_min) + 1 + pad*2).astype(int)

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
    avol = volume_to_ndarray(vol)

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
    return np.argwhere(volume_to_ndarray(volume))


def volume_to_bucket_aims(volume):
    return np.argwhere(volume_to_ndarray(volume))


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


def volume_to_mesh(
        volume,
        gaussian_blur_FWWM=0.5,
        threshold_quantile = 0.9,
        mesh_decimation_params=dict(
            reduction_rate=99,
            max_clearance=3,
            max_error=1,
            feature_angle=180),
        mesh_smoothing_params=dict(
            type='lowpass',
            iterations=30,
            rate=0.4,
            # NOT IMPLEMENTED
            # smoothing feature angle (in degrees) below which the vertex is not moved,
            # only for the Laplacian algorithm, between 0 and 180 degree [default=0]
            # laplacian_angle=0,
            # smoothing restoring force for the Simple Spring and Polygon Spring
            # algorithm, between 0 and 1 [default=0.2]
            # polygonspring_force=0.2
        )):
    """Create a mesh from a volume.

    Before calculating the mesh, the volume is first blurred with a gaussian filter and then thresholded.

    Args:
        volume (np.ndarray or aims.Volume): 3D binary image (if not binary, all vxels != 0 are set to 1)
        gaussian_blur_FWWM (numeric) : the gaussian blur filter's full width at half maximum.
        threshold_quantile (float in [0,1]): voxels below this quantile will be set to zero.
        decimation_params (dict, optional): decimation parameters.
            reduction_rate : expected % decimation reduction rate
            max_clearance : maximum clearance of the decimation
            max_error : maximum error distance from the original mesh (mm)
            feature_angle : feature angle (degrees), between 0 and 180 
        smoothing_params (dict, optional): smoothing parameters
            type : smoothing alorithm's type (laplacian, simplespring, polygonspring or lowpass)
            iterations : smoothing number of iterations
            rate : smoothing moving rate at each iteration

    Returns:
        aims Mesh: A mesh obtained from the input volume
    """

    volume = volume_to_ndarray(volume).astype(float)
    shape = volume.shape

    # gaussiam blur
    sigma = gaussian_blur_FWWM/2.3548200450309493
    volume = gaussian_filter(volume, sigma)
    # normalization
    assert(volume.max()-volume.min() != 0)
    normalize = lambda x : (x-x.min())/(x.max()-x.min())
    volume = normalize(volume)
    # threshold 
    q = np.quantile(volume[volume>0], threshold_quantile)
    volume = (volume > q).astype(int)
    
    # add a -1 pad required for successive steps
    # create an aims volume to use the fillBorder function
    aims_volume = aims.Volume_S16(*shape, 1, 1)
    aims_volume[:, :, :, 0] = (265*volume).astype(np.int16)  # copy data
    aims_volume.fillBorder(-1)

    m = aimsalgo.Mesher()

    smoothing_types = {
        "lowpass": m.LOWPASS,
        'laplacian': m.LAPLACIAN,
        "simplespring": m.SIMPLESPRING,
        "polygonspring": m.POLYGONSPRING
    }

    assert mesh_smoothing_params['type'].lower() in smoothing_types,\
        "smoothing_param must be one of {}, got '{}'".format(
            smoothing_types.keys(), mesh_smoothing_params['type'])

    d = mesh_decimation_params
    m.setDecimation(d['reduction_rate'], d['max_clearance'],
                    d['max_error'], d['feature_angle'])

    d = mesh_smoothing_params
    m.setSmoothing(smoothing_types[d['type']], d['iterations'], d['rate'])

    mesh = aims.AimsTimeSurface()
    m.getBrain(aims_volume, mesh)

    return mesh


def bucket_to_mesh(
        bucket,
        gaussian_blur_FWWM=0.5,
        threshold_quantile = 0.9,
        decimation_params=dict(
            reduction_rate=99,
            max_clearance=3,
            max_error=1,
            feature_angle=180),
        smoothing_params=dict(
            type='lowpass',
            iterations=30,
            rate=0.4,
            # NOT IMPLEMENTED
            # smoothing feature angle (in degrees) below which the vertex is not moved,
            # only for the Laplacian algorithm, between 0 and 180 degree [default=0]
            # laplacian_angle=0,
            # smoothing restoring force for the Simple Spring and Polygon Spring
            # algorithm, between 0 and 1 [default=0.2]
            # polygonspring_force=0.2
        )):
    """Create a mesh from a bucket

    Args:
        bucket (np.ndarray or aims.Volume): sequence of 3D points. 
        gaussian_blur_FWWM (numeric) : the gaussian blur filter's full with at half maximum.
        threshold_quantile (float in [0,1]): voxels below this quantile will be set to zero.
        decimation_params (dict, optional): decimation parameters.
            reduction_rate : expected % decimation reduction rate
            max_clearance : maximum clearance of the decimation
            max_error : maximum error distance from the original mesh (mm)
            feature_angle : feature angle (degrees), between 0 and 180 
        smoothing_params (dict, optional): smoothing parameters
            type : smoothing alorithm's type (laplacian, simplespring, polygonspring or lowpass)
            iterations : smoothing number of iterations
            rate : smoothing moving rate at each iteration

    Returns:
        aims Mesh: A mesh obtained from the input volume
    """
    if not isinstance(bucket, np.ndarray):
        bucket = bucket_aims_to_ndarray(bucket)

    volume = bucket_numpy_to_volume_numpy(bucket)

    return volume_to_mesh(volume, gaussian_blur_FWWM, threshold_quantile, decimation_params, smoothing_params)

# THE OLD IMPLEMENTATION WITH COMMAND LINE TOOLS
# def build_mesh(volume,
#     aimsThreshold = 0,  # aimsThreshold param, smoothing threshold  
#     smoothingFactor = 2.0,  # the smoothing factor used before making MA mesh in step composeSpamMesh 
#     ):

#     tempfile.mkdtemp()
    
#     v = volume
#     # normalize and transform to int16
#     v = ((v - v.min())/(v.max()-v.min())*256).astype(np.float)

#     aims.write(cld.aims_tools.ndarray_to_aims_volume(v), f"{dirpath}/temp.ima")
#     gaussianSmoothCmd = f'AimsGaussianSmoothing -i {dirpath}/temp.ima  -o {dirpath}/temp_smooth.ima -x {smoothingFactor} -y {smoothingFactor} -z {smoothingFactor}'
#     thresholdCmd = f"AimsThreshold -i {dirpath}/temp_smooth.ima -o {dirpath}/temp_threshold.ima -b -t {aimsThreshold}"
#     meshCmd = f'AimsMesh -i {dirpath}/temp_threshold.ima -o {dirpath}/temp.mesh --deciMaxError 0.5 --deciMaxClearance 1 --smooth --smoothIt 20'
#     zcatCmd = f'AimsZCat  -i  {dirpath}/temp*.mesh -o {dirpath}/combined.mesh'

#     sh(gaussianSmoothCmd)
#     sh(thresholdCmd)
#     sh(meshCmd)
#     sh(zcatCmd)

#     return aims.read("tmp/combined.mesh")

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
