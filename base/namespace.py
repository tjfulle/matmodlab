class Namespace(object):
    def __repr__(self):
        string = ", ".join("{0}={1}".format(k, repr(v)) for (k, v) in
                           self.__dict__.items())
        return "Namespace({0})".format(string)
    def items(self):
        return self.__dict__.items()
