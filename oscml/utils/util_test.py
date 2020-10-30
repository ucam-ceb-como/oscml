import logging
import logging.config
import yaml

import oscml.utils.util

def log_fct(logger):
    logger.debug('1 debug message')
    logger.info('2 info message')
    logger.warning('3 warning message')
    logger.error('4 error message')
    logger.critical('5 critical message')

def log_fct_raising_exception(logger):
    logger.foo
    
def test_log():
    loggingconfigfile = './python/conf/logging.yaml'
    oscml.utils.util.init_logging(loggingconfigfile)

    logger = logging.getLogger()
    log_fct(logger)

    # logging exceptions
    try:
        log_fct_raising_exception(logger)
    except Exception:
        # logs exception with stack trace
        logger.exception('6 error message', exc_info=True)

    logger.info('7 just another message')