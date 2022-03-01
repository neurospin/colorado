import logging as _logging
import functools

_log = _logging.getLogger(__name__)

has_aims = None
has_dtb = None
has_brainvisa_and_dico_toolbox = None

# check availability of pyAims from Brainvisa
try:
    from soma import aims
    has_aims = True
except ImportError:
    # aims is not available
    has_aims = False
    _log.info("pyAIMS was not detected.")

# check availability of dico_toolbox (https://github.com/neurospin/dico_toolbox)
try:    
    import dico_toolbox
    has_dtb = True
except ImportError:
    # aims is not available
    has_dtb = False
    _log.info("Dico-toolbox was not detected.")

def with_brainvisa(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):      
        if not has_aims:
            raise RuntimeError(
                "This function is only available in a brainvisa environment")
        return fun(*args, **kwargs)
    return wrapper

has_brainvisa_and_dico_toolbox = has_aims and has_dtb

def with_dico_toolbox(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):      
        if not has_dtb:
            raise RuntimeError(
                "This function is only available with dico_toolbox installed")
        return fun(*args, **kwargs)
    return wrapper

def with_brainvisa_and_dico_toolbox(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):      
        if not (has_aims and has_dtb):
            raise RuntimeError(
                "This function is only available with dico_toolbox installed in a Brainvisa environment")
        return fun(*args, **kwargs)
    return wrapper