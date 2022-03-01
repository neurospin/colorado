import numpy
import plotly.graph_objects as go
import logging
from ..config import has_brainvisa_and_dico_toolbox as _hbv
log = logging.getLogger(__name__)
from .point_cloud import get_point_cloud_g_o

if _hbv:
    from dico_toolbox.convert import bucket_aims_to_ndarray, bucketMAP_aims_to_ndarray


def get_aims_bucket_g_o(aims_bucket, shift=(0, 0, 0), **kwargs):
    """Plot a soma.aims Bucket"""
    log.debug(
        "When drawing an aims bucket, the voxel size is not considered. Draw its BucketMap instead.")
    bucket = bucket_aims_to_ndarray(aims_bucket) + shift
    return get_point_cloud_g_o(bucket, shift=shift, **kwargs)


def get_aims_bucket_map_g_o(aims_bucket_map, shift=(0, 0, 0), **kwargs):
    """Draw thea bucketMap object, which is obtained with aims.read() on .bck files
    The coordinates are scaled according to the voxel size of the MAP."""
    if len(aims_bucket_map) > 1:
        log.warining(
            "The bucketMAP contains more than one buckets. Only the first will be drawn.")
    bck = bucketMAP_aims_to_ndarray(aims_bucket_map)
    return get_point_cloud_g_o(bck, shift=shift, **kwargs)
