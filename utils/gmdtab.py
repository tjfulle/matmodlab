import os
import sys
import time
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesNSImpl


U_ROOT = u"GMDTabular"
U_RUNID = u"runid"
U_DATE = u"date"
U_EVAL = u"Evaluation"
U_EVAL_N = u"n"
U_EVAL_S = u"status"
U_PARAMS = u"Parameters"
U_RESP = u"Responses"

class GMDTabularWriter(object):

    def __init__(self, filepath, runid):
        """
        Set up a logger object, which takes SAX events and outputs
        an XML log file
        """
        if filepath == 1:
            stream = sys.stdout
        else:
            stream = open(filepath, "w")

        encoding = "utf-8"
        self._logger = XMLGenerator(stream, encoding)
        self._logger.startDocument()

        self._stream = stream
        self._encoding = encoding
        self._filepath = filepath

        now = time.asctime(time.localtime())
        vals = {(None, U_DATE): now, (None, U_RUNID): runid}
        names = {(None, U_DATE): U_DATE, (None, U_RUNID): U_RUNID}
        attrs = AttributesNSImpl(vals, names)
        self._logger.startElementNS((None, U_ROOT), U_ROOT, attrs)

    def write_entry(self, eval_number, status, parameters, responses=None):
        """Write an entry

        """
        # Evaluation number
        vals = {(None, U_EVAL_N): str(eval_number),
                (None, U_EVAL_S): str(status)}
        names = {(None, U_EVAL_N): U_EVAL_N, (None, U_EVAL_S): U_EVAL_S}
        attrs = AttributesNSImpl(vals, names)
        self._logger.startElementNS((None, U_EVAL), U_EVAL, attrs)

        # Variables
        vals = {}
        names = {}
        for name, val in parameters.items():
            vals[(None, name)] = str(val)
            names[(None, name)] = name
        attrs = AttributesNSImpl(vals, names)
        self._logger.startElementNS((None, U_PARAMS), U_PARAMS, attrs)
        self._logger.endElementNS((None, U_PARAMS), U_PARAMS)

        # Responses
        if responses:
            vals = {}
            names = {}
            for name, val in responses.items():
                vals[(None, name)] = str(val)
                names[(None, name)] = name
            attrs = AttributesNSImpl(vals, names)
            self._logger.startElementNS((None, U_RESP), U_RESP, attrs)
            self._logger.endElementNS((None, U_RESP), U_RESP)

        #self._logger.characters(msg)
        self._logger.endElementNS((None, U_EVAL), U_EVAL)
        return

    def close(self):
        """
        Clean up the logger object
        """
        self._logger.endElementNS((None, U_ROOT), U_ROOT)
        self._logger.endDocument()
        return


def read_gmd_index(filepath):
    """Read the GMD tabular file

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
        sources.append(os.path.join(dirname, "eval_{0}/{1}.exo".format(n, runid)))
        assert os.path.isfile(sources[-1])
        params = evaluation.getElementsByTagName(U_PARAMS)[0]
        evars, enames = [], []
        for i in range(params.attributes.length):
            attr = params.attributes.item(i)
            enames.append(attr.name)
            evars.append(float(attr.value))
        parameters.append(", ".join("{0}={1:.2g}".format(a, float(b))
                                    for (a, b) in zip(enames, evars)))

    return sources, parameters


if __name__ == "__main__":
    #Test it out
    xl = GMDTabularWriter(1, "foo")
    parameters = {"K": 23, "G": 12}
    i = 0
    for i in range(3):
        parameters["K"] += 1
        parameters["G"] += 1
        xl.write_entry(i, i, parameters)
    xl.close()
