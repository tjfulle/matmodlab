from __config__ import cfg
class Error1(Exception):
    def __init__(self, message):
        if cfg.debug:
            raise Exception("*** gmd: error: {0}".format(message))
        raise SystemExit("*** gmd: error: {0}".format(message))
