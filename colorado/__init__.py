from . import config as _config # this import must remain the first one since it checks if pyAIMS and dico_toolbox are available
from .modules.aims_drawing import anatomist_snatpshot
from .draw import draw, new_figure
from .info import __version__
from . import drawables