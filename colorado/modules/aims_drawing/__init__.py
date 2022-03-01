from colorado.graphic_objects.aims_volume import get_aims_volume_g_o
from ...graphic_objects import *
import logging
from re import match
from ...config import has_brainvisa_and_dico_toolbox as _hbv

if _hbv:
    from soma import aims
    from .anatomist_tools import anatomist_snatpshot as _anatomist_snapshot

log = logging.getLogger(__name__)

drawing_functions = {}
volume_function = None


if _hbv:
    volume_function = get_aims_volume_g_o

    drawing_functions = {
        aims.AimsTimeSurface_3_VOID: get_aims_mesh_g_o,
        aims.BucketMap_VOID.Bucket: get_aims_bucket_g_o,
        aims.Volume_S16: volume_function,
        aims.rc_ptr_Volume_S16: volume_function,
        aims.rc_ptr_Volume_FLOAT: volume_function,
        aims.BucketMap_VOID: get_aims_bucket_map_g_o,
        aims.rc_ptr_BucketMap_VOID: get_aims_bucket_map_g_o
    }


def is_aims_volume(obj):
    """Check wether the type of obj matches the aims Volume od AimsData type."""
    return match(r".*soma.aims.Volume.*", str(type(obj))) or match(r".*soma.aims.AimsData.*", str(type(obj)))


def anatomist_snatpshot(window):
    if has_aims:
        return _anatomist_snapshot(window)
    else:
        log.exception(
            "This function requires Anatomist from the Brainvisa suite.")
