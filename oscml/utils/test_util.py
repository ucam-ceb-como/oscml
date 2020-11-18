import logging
import logging.config
import unittest
import yaml

import oscml.utils.util

class TestModels(unittest.TestCase):

    def log_fct(self, logger):
        logger.debug('1 debug message')
        logger.info('2 info message')
        logger.warning('3 warning message')
        logger.error('4 error message')
        logger.critical('5 critical message')

    def log_fct_raising_exception(self, logger):
        logger.foo
        
    def test_log(self):
        oscml.utils.util.init_logging('.', './tmp')

        logger = logging.getLogger()
        self.log_fct(logger)

        # logging exceptions
        try:
            self.log_fct_raising_exception(logger)
        except Exception:
            # logs exception with stack trace
            logger.exception('6 error message', exc_info=True)

        logger.info('7 just another message')

if __name__ == '__main__':
    unittest.main()