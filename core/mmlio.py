import sys
from core.exomgr import ExodusIIManager
from core.logger import Logger

exo = ExodusIIManager()
logger = Logger()

def log_warning(msg):
    logger.warn(msg)

def log_error(msg):
    logger.error(msg)

def log_message(msg):
    logger.write(msg)

def Error1(msg):
    log_error(msg)
