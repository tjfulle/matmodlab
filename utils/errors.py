class Error1(Exception):
    def __init__(self, message):
        raise SystemExit("*** gmd: error: {0}".format(message))
