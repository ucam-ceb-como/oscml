"""Root package info."""

__version__ = '0.0.5'
__author__ = 'Dr. Andreas Eibeck et al.'
__author_email__ = ''
__license__ = 'Apache-2.0'
__copyright__ = 'Copyright (c) 2020-2021, %s.' % __author__
__homepage__ = ''
__docs__ = (
    ""
)
__long_docs__ = """
"""

import oscml.utils.util
from oscml.utils.util import log

def init():
    loggingconfigfile = './conf/logging.yaml'
    oscml.utils.util.init_logging(loggingconfigfile)
    log('initializing finished, version=', __version__)
    
print('__init__.py is called')
init()