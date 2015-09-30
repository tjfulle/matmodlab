import logging
from ..product import SPLASH
from ..mml_siteenv import environ

# Monkey path the logging stream handler emit function
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
def emit(self, record):
    '''Monkey-patch the logging StreamHandler emit function. Allows omiting
    trailing newline when not wanted'''
    if hasattr(self, 'baseFilename'):
        fs = '%s\n'
    else:
        fs = '%s' if getattr(record, 'continued', False) else '%s\n'
    self.stream.write(fs % self.format(record))
    self.flush()
logging.StreamHandler.emit = emit

def setup_logger(name, filename=None, verbosity=None, splashed=[0]):
    """Set up the logger"""

    if environ.notebook:
        level = logging.WARNING

    elif environ.parent_process:
        level = logging.CRITICAL

    elif verbosity is not None:
        environ.log_level = verbosity
        level = environ.log_level

    else:
        level = environ.log_level

    logger = logging.getLogger(name)
    logger.propagate = False
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)

    if filename is not None:
        fh = logging.FileHandler(filename, mode='w')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    if not splashed[0]:
        logger.info(SPLASH)
        splashed[0] += 1
    elif filename is not None:
        fh.stream.write(SPLASH)

    return logger
