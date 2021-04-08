import tempfile
from PIL import Image
from numpy import asarray
from numpy.lib.arraysetops import isin
import anatomist

def anatomist_snatpshot(window):
    """Return a screenshot of an anatomist window as numpy array that can be plotted.

    Args:
        window (anatomist.cpp.weak_shared_ptr_AWindow: a reference to an open anatpmist window

    Returns:
        numpy.ndarray: A snapshot of the window
    """

    with tempfile.NamedTemporaryFile(suffix='_temp.jpg', prefix='pyanatomist_') as f:
        window.snapshot(f.name)
        img = asarray(Image.open(f.name))
        return img
