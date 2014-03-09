import sys
from select import select

def timed_raw_input(message, timeout=10, default=None):
    # from stackoverflow.com/questions/3471461/raw-input-and-timeout
    sys.stdout.write(message)
    sys.stdout.flush()
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.readline()
    else:
        return default
