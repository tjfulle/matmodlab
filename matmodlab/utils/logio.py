import logging
from matmodlab.product import SPLASH
from matmodlab.mml_siteenv import environ

def setup_logger(name, filename, verbosity=None, splashed=[0]):
    """Set up the logger"""

    if environ.parent_process or environ.notebook:
        level = logging.CRITICAL

    elif environ.notebook:
        level = logging.WARNING

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

    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    if not splashed[0]:
        logger.info(SPLASH)
        splashed[0] += 1
    else:
        fh.stream.write(SPLASH)

    return logger
