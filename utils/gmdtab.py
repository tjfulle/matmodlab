import os
import sys
import time
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesNSImpl
import xml.dom.minidom as xdom


F_EVALDB = "gmd-evaldb.xml"
U_ROOT = u"GMDTabular"
U_RUNID = u"runid"
U_DATE = u"date"
U_EVAL = u"Evaluation"
U_EVAL_N = u"n"
U_EVAL_D = u"d"
U_EVAL_S = u"status"
U_PARAMS = u"Parameters"
U_RESP = u"Responses"
IND = "  "


class GMDTabularWriter(object):

    def __init__(self, runid, d=None):
        """Set up a logger object, which takes evaluation events and outputs
        an XML log file

        """
        self.runid = runid
        self.stack = []

        if d is None:
            self._stream = sys.stdout
            self._filepath = None
        else:
            self._filepath = os.path.join(d, F_EVALDB)
            self._stream = open(self._filepath, "w")

        self.start_document()
        pass

    def create_element(self, name, attrs):
        sp = IND * len(self.stack)
        a = " ".join('{0}="{1}"'.format(k, v) for (k, v) in attrs)
        self._stream.write("{0}<{1} {2}/>\n".format(sp, name, a))
        return

    def start_element(self, name, attrs, end=False):
        sp = IND * len(self.stack)
        a = " ".join('{0}="{1}"'.format(k, v) for (k, v) in attrs)
        self._stream.write("{0}<{1} {2}>\n".format(sp, name, a))
        self.stack.append(name)
        return

    def end_element(self, name):
        _name = self.stack.pop(-1)
        assert _name == name
        sp = IND * len(self.stack)
        self._stream.write("{0}</{1}>\n".format(sp, name))
        return

    def start_document(self):
        self._stream.write("""<?xml version="1.0"?>\n""")
        now = time.asctime(time.localtime())
        self.start_element(U_ROOT, (("runid", self.runid), ("date", now)))
        return

    def end_document(self):
        _name = self.stack.pop(-1)
        assert _name == U_ROOT
        self._stream.write("</{0}>\n".format(U_ROOT))
        self._stream.flush()
        self._stream.close()
        return

    def write_eval_info(self, n, s, d, parameters, responses=None):
        """Write information for this evaluation

        Parameters
        ----------
        n : int
            Evaluation number
        s : int
            Evaluation status
        d : int
            Evaluation directory
        parameters : list of tuple
            (name, value) pairs for each parameter
        respones : list of tuple (optional)
            (name, value) pairs for each response

        """
        self.start_element(U_EVAL, ((U_EVAL_N, n), (U_EVAL_S, s), (U_EVAL_D, d)))
        self.create_element(U_PARAMS, parameters)
        if responses:
            self.create_element(U_RESP, responses)
        self.end_element(U_EVAL)
        return

    def close(self):
        """
        Clean up the logger object
        """
        self.end_document()
        return


def read_gmd_evaldb(filepath):
    """Read the GMD tabular file

    Parameters
    ----------
    filepath : str
        Path to index file to read

    Returns
    -------
    sources : list of str
        Individual filepaths for each evaluation
    parameters : tuple of tuple
        (name, value) pairs for parameters for each evaluation

    """
    import xml.dom.minidom as xdom
    dirname = os.path.realpath(os.path.dirname(filepath))
    doc = xdom.parse(filepath)
    root = doc.getElementsByTagName(U_ROOT)[0]
    runid = root.getAttribute("runid")

    sources = []
    parameters = []
    for evaluation in root.getElementsByTagName(U_EVAL):
        n = evaluation.getAttribute(U_EVAL_N)
        d = evaluation.getAttribute(U_EVAL_D)
        sources.append(os.path.join(d, "{0}.exo".format(runid)))
        assert os.path.isfile(sources[-1])
        params = evaluation.getElementsByTagName(U_PARAMS)[0]
        evars, enames = [], []
        for i in range(params.attributes.length):
            attr = params.attributes.item(i)
            enames.append(attr.name)
            evars.append(float(attr.value))
        parameters.append(zip(enames, evars))
    return sources, parameters


if __name__ == "__main__":
    #Test it out
    xl = GMDTabularWriter(1, "foo")
    parameters = [["K", 23], ["G", 12]]
    i = 0
    for i in range(3):
        parameters[0][1] += 1
        parameters[1][1] += 1
        xl.write_entry(i, i, i, parameters)
    xl.close()
