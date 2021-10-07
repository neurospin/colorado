# from scipy.ndimage.measurements import minimum
import os
import tempfile
from soma import aims, aimsalgo
import numpy as np
from scipy.ndimage import gaussian_filter

import logging
log = logging.getLogger(__name__)


def ndarray_to_volume_aims(ndarray):
    """Create a new volume from the numpy ndarray.
    The array is first converted to Fortran ordering,
    as requested by the Volume constructor

    :param ndarray: the volume data
    :type ndarray: numpy.ndarray
    :return: an Aims Volume object containing the same data as ndarray
    :rtype: aims.Volume
    """
    ndarray.reshape(*ndarray.shape, 1)
    return aims.Volume(np.asfortranarray(ndarray))


ndarray_to_aims_volume = ndarray_to_volume_aims


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

    if aims_bucket.size() > 0:
        v = np.empty((aims_bucket.size(), len(aims_bucket.keys()[0].list())))
        for i, point in enumerate(aims_bucket.keys()):
            v[i] = point.arraydata()
    else:
        log.warning("Empty bucket! This can be a source of problems...")
        v = np.empty(0)
    return v


def bucket_numpy_to_bucket_aims(ndarray):
    """Transform a (N,3) ndarray into an aims BucketMap_VOID.
    The coordinates in the input array are casted to int.
    """

    assert ndarray.shape[1] == 3, " ndarray shape must be (N,3)"

    if ndarray.dtype != int:
        ndarray = ndarray.astype(int)

    # create aims bucketmap instance
    bck = aims.BucketMap_VOID()
    b0 = bck[0]

    # fill the bucket
    for x, y, z in ndarray:
        b0[x, y, z] = 1

    return bck


def volume_to_ndarray(volume):
    """Transform aims volume in numpy array.

    Takes the first element for every dimensions > 3.

    Args:
        volume (aims.volume): aims volume
    """
    # remove all dimensions except the 3 first
    # take element 0 for the others
    try:
        # aims VOlume and numpy array have shape
        if len(volume.shape) > 3:
            volume = volume[tuple(3*[slice(0, None)] + [0]
                                  * (len(volume.shape)-3))]
    except AttributeError:
        # aims.AimsData does not have shape
        # but it is always 3D
        volume = volume[:, :, :, 0]
    return volume[:]


def _volume_size_from_numpy_bucket(bucket_array, pad):
    a = bucket_array
    # the minimum and maximum here make sure that the voxels
    # are in the absolute coordinates system of the bucket
    # i.e. the volume always include the bucket origin.
    # This is the behaviour of AIMS
    # this also makes the volume bigger and full with zeros
    v_max = np.maximum((0, 0, 0), a.max(axis=0))
    v_min = np.minimum((0, 0, 0), a.min(axis=0))
    v_size = np.ceil(abs(v_max - v_min) + 1 + pad*2).astype(int)
    return v_size, v_min


def _point_to_voxel_indices(point):
    """transform the point coordinates into a tuple of integer indices.

    Args:
        point (Sequence[numeric]): point coordinates

    Returns:
        numpy.ndarray of type int: indices
    """
    return np.round(point).astype(int)


def bucket_numpy_to_volume_numpy(bucket_array, pad=0, side=None):
    """Transform a bucket into a 3d boolean volume.
    Input and output types are numpy.ndarray"""

    v_size, v_min = _volume_size_from_numpy_bucket(bucket_array, pad)

    vol = np.zeros(np.array(v_size))

    for p in bucket_array:
        x, y, z = _point_to_voxel_indices((p-v_min)+pad)
        vol[x, y, z] = 1

    return vol


def bucket_numpy_to_volume_aims(bucket_array, pad=0):
    """Transform a bucket into a 3d boolean volume.
    Input and output types are numpy.ndarray"""

    v_size, v_min = _volume_size_from_numpy_bucket(bucket_array, pad)

    vol = aims.Volume(*v_size, dtype='int16')
    vol.fill(0)
    avol = volume_to_ndarray(vol)

    for p in bucket_array:
        x, y, z = _point_to_voxel_indices(p-v_min+pad)
        avol[x, y, z] = 1

    return vol


def bucket_aims_to_volume_aims(aims_bucket, pad=0):
    """Transform a bucket into a 3d boolean volume.
    Input and output types are aims objects"""

    # TODO : transfer metadata
    # e.g. the dxyz is kept in the aims volume

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


def volume_to_mesh(volume, smoothingFactor=2.0, aimsThreshold='96%',
                   deciMaxError=0.5, deciMaxClearance=1.0, smoothIt=20, translation=(0, 0, 0), transl_scale=30):
    """Generate the mesh of the input volume.
    WARNING: This function directly call some BrainVisa command line tools via os.system calls.

    Args:
        volume (nparray or pyaims volume): The input volume.
        smoothingFactor (float, optional): Standard deviation of the 3D isomorph Gaussian filter of the input volume.
        aimsThreshold (float or str) : First threshold value. All voxels below this value are not considered.
            The threshold can be expressed as:
            - a float, representing the threshold intensity
            - a percentage (e.g. "95%") which represents the percentile of low-value pixel to eliminate.
        deciMaxError (float) : Maximum error distance from the original mesh (mm).
        deciMaxClearance (float) : Maximum clearance of the decimation.
        smoothIt (int) : Number of mesh smoothing iteration.
        translation (vector or 3 int) : translation to apply to the calculated mesh

    Returns:
        aims Mesh : the mesh of the inputn volume.
    """

    log.debug("volume_to_mesh:")

    v = volume[:]
    # normalize
    v = ((v - v.min())/(v.max()-v.min())*256).astype(np.float)

    # temporary directory
    dirpath = tempfile.mkdtemp()

    fname = "temp_initial.ima"
    # write volume to file
    aims.write(ndarray_to_aims_volume(v), f"{dirpath}/{fname}")

    # Gaussian blur
    if smoothingFactor > 0:
        out_fname = "temp_smoothed.ima"
        gaussianSmoothCmd = f'AimsGaussianSmoothing -i {dirpath}/{fname} -o {dirpath}/{out_fname} -x {smoothingFactor} -y {smoothingFactor} -z {smoothingFactor}'
        log.debug(gaussianSmoothCmd)
        os.system(gaussianSmoothCmd)
        fname = out_fname

    # Threshold
    # read the blurred volume values and calculate the threshold
    v = aims.read(f"{dirpath}/{fname}")[:]
    nonzero_voxels = v[v > 0].flatten()
    if type(aimsThreshold) == str:
        # the threshold is a string
        if aimsThreshold[-1] == '%':
            # use the percentage value
            q = float(aimsThreshold[:-1])
            aimsThreshold = np.percentile(nonzero_voxels, q)
        else:
            raise ValueError(
                "aimsThreshold must be a float or a string expressing a percentage (eg '90%')")

    out_fname = "temp_threshold.ima"
    thresholdCmd = f"AimsThreshold -i {dirpath}/{fname} -o {dirpath}/{out_fname} -b -t {aimsThreshold}"
    log.debug(thresholdCmd)
    os.system(thresholdCmd)
    fname = out_fname

    # MESH
    # Generate one mesh per interface (connected component?)
    if smoothIt is not None and smoothIt is not 0:
        smooth_arg = f"--smooth --smoothIt {smoothIt}"
    else:
        smooth_arg = ""

    meshCmd = f'AimsMesh -i {dirpath}/{fname} -o {dirpath}/temp.mesh --decimation --deciMaxError {deciMaxError} --deciMaxClearance {deciMaxClearance} {smooth_arg}'
    log.debug(meshCmd)
    os.system(meshCmd)

    # Concatenate the meshes
    zcatCmd = f'AimsZCat  -i  {dirpath}/temp*.mesh -o {dirpath}/combined.mesh'
    log.debug(zcatCmd)
    os.system(zcatCmd)

    mesh = aims.read(f"{dirpath}/combined.mesh")

    assert len(translation) == 3, "len(translation) must be 3"

    for i in range(3):
        # for each dimension
        if translation[i] != 0:
            mesh = shift_aims_mesh(
                mesh, translation[i], scale=transl_scale, axis=i)

    return mesh


# def volume_to_mesh_experimental(
#         volume,
#         gaussian_blur_FWWM=0,
#         threshold_absolute=None,
#         threshold_quantile=0,
#         translation=(0, 0, 0),
#         decimation_params=dict(
#             reduction_rate=99,
#             max_clearance=3,
#             max_error=1,
#             feature_angle=180),
#         smoothing_params=dict(
#             algorithm='lowpass',
#             iterations=30,
#             rate=0.4,
#             # NOT IMPLEMENTED
#             # smoothing feature angle (in degrees) below which the vertex is not moved,
#             # only for the Laplacian algorithm, between 0 and 180 degree [default=0]
#             # laplacian_angle=0,
#             # smoothing restoring force for the Simple Spring and Polygon Spring
#             # algorithm, between 0 and 1 [default=0.2]
#             # polygonspring_force=0.2
#         )):
#     """Create a mesh from a volume.

#     Before calculating the mesh, the volume is first blurred with a gaussian filter and then thresholded.

#     Args:
#         volume (np.ndarray or aims.Volume): 3D binary image (if not binary, all vxels != 0 are set to 1)
#         gaussian_blur_FWWM (numeric) : the gaussian blur filter's full width at half maximum.
#         threshold_absolute all voxels lower than this value will be set to zero.
#         threshold_quantile (float in [0,1]): voxels with intensity below this quantile will be set to zero.
#             Applied only if threshold_absolute is None.
#         decimation_params (dict, optional): decimation parameters.
#             reduction_rate : expected % decimation reduction rate
#             max_clearance : maximum clearance of the decimation
#             max_error : maximum error distance from the original mesh (mm)
#             feature_angle : feature angle (degrees), between 0 and 180
#         smoothing_params (dict, optional): smoothing parameters
#             type : smoothing alorithm's type (laplacian, simplespring, polygonspring or lowpass)
#             iterations : smoothing number of iterations
#             rate : smoothing moving rate at each iteration

#     Returns:
#         aims Mesh: A mesh obtained from the input volume
#     """

#     log.debug("Volume to mesh:")
#     volume = volume_to_ndarray(volume).astype(float)
#     shape = volume.shape

#     # gaussiam blur
#     log.debug(" - Gaussian blur")
#     if gaussian_blur_FWWM > 0:
#         sigma = gaussian_blur_FWWM/2.3548200450309493
#         volume = gaussian_filter(volume, sigma)

#     # normalization
#     log.debug(" - Normalization")
#     assert(volume.max()-volume.min() != 0)
#     def normalize(x): return (x-x.min())/(x.max()-x.min())
#     volume = normalize(volume)

#     # threshold
#     log.debug(" - threshold")
#     if threshold_absolute is not None:
#         q = threshold_absolute
#     elif threshold_quantile > 0:
#         q = np.quantile(volume[volume > 0], threshold_quantile)
#     else:
#         q = 0
#     volume = (volume > q).astype(int)

#     log.debug(" - add -1 border")
#     # add a -1 pad required for successive steps
#     # create an aims volume to use the fillBorder function
#     aims_volume = aims.Volume_S16(*shape, 1, 1)
#     aims_volume[:, :, :, 0] = (265*volume).astype(np.int16)  # copy data
#     aims_volume.fillBorder(-1)

#     m = aimsalgo.Mesher()

#     # Ça renvoie un dict (label: liste de maillages, un par interface).
#     mesher_output = m.doit(aims_volume)
#     # WARGNING: the following assumes that mesher_output contains only one interface.
#     # Merge the meshes
#     meshes = list(mesher_output.values())[0]
#     merged_mesh = meshes[0]
#     for mesh in meshes[1:]:
#         aims.SurfaceManip.meshMerge(merged_mesh, mesh)

#     mesh = merged_mesh

#     if mesh.size() == 0:
#         raise Exception("The mesh is empty, try different parameters")

#     # DECIMATE
#     if decimation_params is not None:
#         d = decimation_params
#         print(d)
#         m.setDecimation(d['reduction_rate'], d['max_clearance'],
#                         d['max_error'], d['feature_angle'])

#         m.decimate(mesh)

#     # SMOOTH
#     if smoothing_params is not None:
#         smoothing_types = {
#             "lowpass": m.LOWPASS,
#             'laplacian': m.LAPLACIAN,
#             "simplespring": m.SIMPLESPRING,
#             "polygonspring": m.POLYGONSPRING
#         }
#         assert smoothing_params['algorithm'].lower() in smoothing_types,\
#             "smoothing_param must be one of {}, got '{}'".format(
#             smoothing_types.keys(), smoothing_params['algorithm'])

#         d = smoothing_params
#         m.setSmoothing(
#             smoothing_types[d['algorithm']], d['iterations'], d['rate'])

#         m.smooth(mesh)

#     if translation != (0, 0, 0):
#         mesh = PyMesh(mesh)
#         # add translation
#         mesh.vertices = mesh.vertices + translation
#         mesh = mesh.to_aims_mesh()

#     return mesh


def bucket_to_mesh(bucket, smoothingFactor=0, aimsThreshold=1,
                   deciMaxError=0.5, deciMaxClearance=1.0, smoothIt=20, translation=(0, 0, 0), scale=30):
    """Generate the mesh of the input bucket.
    WARNING: This function directly call some BrainVisa command line tools via os.system calls.

    Args:
        bucket (nparray or pyaims bucket): The input bucket.
        smoothingFactor (float, optional): Standard deviation of the 3D isomorph Gaussian filter of the input volume.
        aimsThreshold (float or str) : First threshold value. All voxels below this value are not considered.
        deciMaxError (float) : Maximum error distance from the original mesh (mm).
        deciMaxClearance (float) : Maximum clearance of the decimation.
        smoothIt (int) : Number of mesh smoothing iteration.
        translation (vector or 3 int) : translation to apply to the calculated mesh

    Returns:
        aims Mesh : the mesh of the inputn volume.
    """

    if isinstance(bucket, aims.BucketMap_VOID.Bucket):
        bucket = bucket_aims_to_ndarray(bucket)
    elif isinstance(bucket, aims.BucketMap_VOID):
        raise ValueError("Input is a BucketMap, not a bucket.")

    if any([x-int(x) != 0 for x in bucket[:].ravel()]):
        log.warn(
            "This bucket's coordinates are not integers. Did you apply any transformation to it?")

    # x, y, z = bucket.T
    # translation = (x.min(), y.min(), z.min())

    volume = bucket_numpy_to_volume_numpy(bucket)

    return volume_to_mesh(volume, smoothingFactor=smoothingFactor, aimsThreshold=aimsThreshold,
                          deciMaxError=deciMaxError, deciMaxClearance=deciMaxClearance, smoothIt=smoothIt,
                          translation=translation, transl_scale=scale)


def buket_to_aligned_mesh(*args, **kwargs):
    raise SyntaxError(
        "This function is deprecated due to misspelling of 'bucket', please use bucket_to_aligned_mesh")


def bucket_to_aligned_mesh(raw_bucket, talairach_dxyz, talairach_rot, talairach_tr, align_rot, align_tr, flip=False, **kwargs):
    """Generate the mesh of the given bucket.

    The mesh is transformed according to the given rotations and translations.

    The Talairach parameters are the scaling vector, the rotation matrix and the translation vector of the Talairach transform.
    The align paramenters are the rotation matrix and translation vector of the alignment with the central subjet.

    The kwargs are directly passed to cld.aims_tools.bucket_to_mesh().
    """

    # Generate mesh
    mesh = bucket_to_mesh(raw_bucket, **kwargs)

    dxyz = talairach_dxyz.copy()

    # Rescale mesh
    rescale_mesh(mesh, dxyz)

    # apply Talairach transform
    M1 = get_aims_affine_transform(talairach_rot, talairach_tr)
    aims.SurfaceManip.meshTransform(mesh, M1)

    if flip:
        flip_mesh(mesh)

    # apply alignment transform
    M2 = get_aims_affine_transform(align_rot, align_tr)
    aims.SurfaceManip.meshTransform(mesh, M2)

    return mesh

# def bucket_to_mesh_experimental(
#         bucket,
#         gaussian_blur_FWWM=0,
#         threshold_absolute=None,
#         threshold_quantile=0,
#         decimation_params=dict(
#             reduction_rate=99,
#             max_clearance=3,
#             max_error=1,
#             feature_angle=180),
#         smoothing_params=dict(
#             algorithm='lowpass',
#             iterations=30,
#             rate=0.4,
#             # NOT IMPLEMENTED
#             # smoothing feature angle (in degrees) below which the vertex is not moved,
#             # only for the Laplacian algorithm, between 0 and 180 degree [default=0]
#             # laplacian_angle=0,
#             # smoothing restoring force for the Simple Spring and Polygon Spring
#             # algorithm, between 0 and 1 [default=0.2]
#             # polygonspring_force=0.2
#         )):
#     """Create a mesh from a bucket

#     Args:
#         bucket (np.ndarray or aims.Volume): sequence of 3D points.
#         gaussian_blur_FWWM (numeric) : the gaussian blur filter's full with at half maximum.
#         threshold_absolute all voxels lower than this value will be set to zero.
#         threshold_quantile (float in [0,1]): voxels with intensity below this quantile will be set to zero.
#             Applied only if threshold_absolute is None.
#         decimation_params (dict, optional): decimation parameters.
#             reduction_rate : expected % decimation reduction rate
#             max_clearance : maximum clearance of the decimation
#             max_error : maximum error distance from the original mesh (mm)
#             feature_angle : feature angle (degrees), between 0 and 180
#         smoothing_params (dict, optional): smoothing parameters
#             type : smoothing alorithm's type (laplacian, simplespring, polygonspring or lowpass)
#             iterations : smoothing number of iterations
#             rate : smoothing moving rate at each iteration

#     Returns:
#         aims Mesh: A mesh obtained from the input volume
#     """
#     if not isinstance(bucket, np.ndarray):
#         bucket = bucket_aims_to_ndarray(bucket)

#     x, y, z = bucket.T
#     translation = (x.min(), y.min(), z.min())

#     volume = bucket_numpy_to_volume_numpy(bucket)

#     if gaussian_blur_FWWM == 0 and threshold_quantile != 0:
#         log.warn("Thresholding is automatically disabled with smoothing FWHM=0. To remove this message set threshold_quantile=0")
#         threshold_quantile = 0

#     return volume_to_mesh_experimental(volume, gaussian_blur_FWWM, threshold_absolute, threshold_quantile, translation, decimation_params, smoothing_params)


def get_aims_affine_transform(rotation_matrix, transltion_vector):
    """Get an aims AffineTransformation3d from rotation matrix and rotation vector"""
    m = np.hstack([rotation_matrix, transltion_vector.reshape(-1, 1)])
    M = aims.AffineTransformation3d()
    M.fromMatrix(m)
    return M


def rescale_mesh(mesh, dxyz):
    """Rescale a mesh by multiplying its vertices with the factors in dxyx.
    The rescaling is done in place."""
    for i in range(mesh.size()):
        mesh.vertex(i).assign(
            [aims.Point3df(np.array(x[:])*dxyz) for x in mesh.vertex(i)])


def flip_mesh(mesh, axis=0):
    """Flip the mesh by inverting the specified axis"""
    flip_v = np.ones(3)
    flip_v[axis] = -1
    for i in range(mesh.size()):
        mesh.vertex(i).assign(
            [aims.Point3df(np.array(x[:])*flip_v) for x in mesh.vertex(i)])


def shift_aims_mesh(mesh, offset, scale=1):
    """Translate each mesh of a specified distance along an axis.

    The scale parameter multiplies the distance values before applying the translation.
    Returns a shifted mesh
    """
    try:
        iter(offset)
    except TypeError:
        raise TypeError(
            "Offset must be an iterable of length 3. Use shift_aims_mesh_along_axis() to apply a scalar offset to a given axis")

    if len(offset) != 3:
        raise ValueError("len(offset) must be 3.")

    offset_mesh = aims.AimsTimeSurface(mesh)
    vertices = np.array([x[:] for x in mesh.vertex(0)])
    for axis in range(3):
        vertices[:, axis] += offset[axis]*scale
    offset_mesh.vertex(0).assign(vertices.tolist())
    return offset_mesh


def shift_aims_mesh_along_axis(mesh, offset, scale=30, axis=1):
    shift_v = np.zeros(3)
    shift_v[axis] = offset
    return shift_aims_mesh(mesh, shift_v, scale=scale)


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

    @ property
    def vertices(self):
        return self.frames[0].vertices

    @ vertices.setter
    def vertices(self, v):
        self.frames[0].vertices = v

    @ property
    def polygons(self):
        return self.frames[0].polygons

    @ polygons.setter
    def polygons(self, v):
        self.frames[0].polygons = v

    @ property
    def normals(self):
        return self.frames[0].normals

    @ normals.setter
    def normals(self, v):
        self.frames[0].normals = v

    def append(self, frame):
        self.frames.append(frame)

    def __getitem__(self, i):
        try:
            return self.frames[i]
        except IndexError:
            raise IndexError("this mesh is empty")

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

    @ staticmethod
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
