import sys
from ..constants import *

def cprint(string):
    sys.stderr.write(string + "\n")

class BreakPointError(Exception):
    def __init__(self, c):
        super(BreakPointError, self).__init__("{0}: not a valid conditon".format(c))

class BreakPointStop(Exception):
    pass

class BreakPoint:
    def __init__(self, condition, mps, xit=0):
        self._condition = condition
        self.mps = mps
        self.xit = xit
        self.condition, self.names = self.parse_condition(condition)
        self.first = 1

    @staticmethod
    def parse_condition(condition):
        """Parse the original condition given in the input file

        """
        from StringIO import StringIO
        from tokenize import generate_tokens
        from token import NUMBER, OP, NAME, ENDMARKER
        wrap = lambda x: "{{{0}}}".format(x).upper()
        s = StringIO(condition)
        g = [x for x in generate_tokens(s.readline)]
        stack = []
        names = []
        expr = []
        for toknum, tokval, _, _, _ in g:
            if len(expr) == 3:
                stack.append(" ".join(expr))
                expr = []
            if toknum == ENDMARKER:
                break
            if not expr:
                if toknum != NAME:
                    raise BreakPointError(condition)
                if tokval in ("and", "or"):
                    stack.append(tokval)
                    continue
                names.append(tokval)
                expr.append(wrap(tokval))
            elif len(expr) == 1:
                if toknum != OP:
                    raise BreakPointError(condition)
                if tokval not in ("==", ">", ">=", "<", "<="):
                    raise BreakPointError(condition)
                expr.append(tokval)
            elif len(expr) == 2:
                if toknum not in (NUMBER, NAME):
                    raise BreakPointError(condition)
                if toknum == NAME:
                    names.append(tokval)
                    tokval = wrap(tokval)
                expr.append(tokval)
        condition = " ".join(stack)
        return condition, names

    def eval(self, frame):
        """Evaluate the break condition

        """
        if not self.condition:
            return

        kwds = {"TIME": frame.value}
        for (name, fo) in frame.field_outputs.items():
            if fo.type == SCALAR:
                kwds[name] = fo.data[0]
            else:
                for (i, key) in enumerate(fo.keys):
                    kwds[key] = fo.data[0, i]
        condition = self.condition.format(**kwds)
        if not eval(condition):
            return

        if self.xit:
            raise BreakPointStop

        # Break condition met.  Enter the UI
        self.ui(condition, frame)

        return

    def generate_summary(self, frame):
        material = self.mps.material
        params = zip(material.parameter_names, material.params)
        params = "\n".join("  {0} = {1}".format(a, b) for a, b in params)
        time = frame.value
        data = []
        for (name, fo) in frame.field_outputs.items():
            if fo.type == SCALAR:
                data.append('  {0} : {1:.4f}'.format(name, fo.data[0]))
            else:
                v = ', '.join('{0:.4f}'.format(_) for _ in fo.data[0])
                data.append('  {0} : [{1}]'.format(name, v))
        data = '\n'.join(data)
        summary = """
SUMMARY OF PARAMETERS
{0}

SUMMARY OF DATA
  TIME : {1:.4f}
{2}

""".format(params, time, data)
        return summary

    def ui(self, condition, frame):
        self.summary = self.generate_summary(frame)

        if self.first:
            cprint(self.manpage(condition, frame.value))
            self.first = 0
        else:
            cprint("BREAK CONDITION {0} ({1}) "
                   "MET AT TIME={2}".format(self._condition, condition, frame.value))

        kwds = {'TIME': frame.value}
        for (name, fo) in frame.field_outputs.items():
            if fo.type == SCALAR:
                kwds[name] = fo.data[0]
            else:
                kwds[name] = fo.data[0]
                for (i, key) in enumerate(fo.keys):
                    kwds[key] = fo.data[0, i]

        while 1:
            resp = raw_input("mml > ").lower().split()

            if not resp:
                continue

            if resp[0] == "c":
                self.condition = None
                return

            if resp[0] == "h":
                cprint(self.manpage(condition, frame.value))
                continue

            if resp[0] == "s":
                return

            if resp[0] == "p":
                toprint = resp[1:]
                if not toprint:
                    cprint(self.summary)
                    continue

                for item in toprint:
                    if item.upper() == "TIME":
                        name = "TIME"
                        value = frame.value
                    elif item[:5] == "param":
                        name = "PARAMETERS"
                        value = self.mps.material.parameters
                    elif item.upper() in data:
                        name = item.upper()
                        value = data[name]
                    elif item.upper() in self.mps.material.parameter_names:
                        name = item.upper()
                        idx = self.mps.material.parameter_names.index(name)
                        value = self.mps.material.params[idx]
                    else:
                        cprint("  {0}: not valid variable".format(item))
                        continue
                    cprint("  {0} = {1}".format(name, value))
                continue

            if resp[0] == "q":
                raise BreakPointStop

            else:
                cprint("{0}: unrecognized command".format(" ".join(resp)))

    def manpage(self, condition, time):
        page = """

BREAK CONDITION {0} ({1}) MET AT TIME={2}

SYNOPSIS OF COMMANDS
    c
      Continue the analysis until completion [break condition removed].

    h
      Print the help message

    p [name_1[ name_2[ ...[name_n]]]]
      Print simulation information to the screen.  If the optional name
      is given, the current value of that variable and/or parameter is
      printed.

    s
      Step through the analysis, reevaluating the break point at each step.

    set <name> <value> [type]
      Set the variable or parameter to the new value.
      In the case that both a variable and parameter have the same name
      specify type=v for variable or type=p for parameter [default].
      Note, this could have unintended consequences on the rest of the
      simulation

    q
      Quit the analysis gracefully.

EXAMPLES
    o Setting the value of the bulk modulus K for the remainder of the analysis

      set K 100
      c

    """.format(self._condition, condition, time)
        return page
