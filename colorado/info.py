# -*- coding: utf-8 -*-

version_major = 0
version_minor = 2
version_micro = 0
version_extra = ''

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
__version__ = "%s.%s.%s%s" % (version_major,
                              version_minor,
                              version_micro,
                              version_extra)
CLASSIFIERS = [
    "Programming Language :: Python",
    "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
    "Operating System :: OS Independent"]


description = "A plotly interface for pyAims (brain images)"

# versions for dependencies
SPHINX_MIN_VERSION = '1.0'

# Main setup parameters
NAME = 'Colorado'
PROJECT = 'colorado'
ORGANISATION = "neurospin"
MAINTAINER = "nobody"
MAINTAINER_EMAIL = "support@neurospin.info"
DESCRIPTION = description
URL = "https://github.com/neurospin/colorado"
DOWNLOAD_URL = "https://github.com/neurospin/colorado"
LICENSE = "CeCILL-B"
AUTHOR = "Marco Pascucci"
AUTHOR_EMAIL = 'marpas.paris@gmail.com'
PLATFORMS = "OS Independent"
PROVIDES = ["colorado"]
REQUIRES = ['plotly', 'numpy', 'dico_toolbox']
EXTRA_REQUIRES = {
    "doc": ["sphinx>=" + SPHINX_MIN_VERSION]}

brainvisa_build_model = 'pure_python'

