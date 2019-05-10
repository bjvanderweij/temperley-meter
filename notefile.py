import re 
from warnings import warn

# Read a notefile, transform into pip grid

def onsets(lines):
    r = re.compile(r'^Note +([0-9]+) +([0-9]+) +([0-9])+ *$')
    onsets = []
    for line in lines:
        m = re.match(r, line)
        if m is None:
            warn('Line not matched: %s' % line)
            continue
        onsets.append(int(m.group(2)))
    return onsets

def pips(onsets, pip=50):
    shift = onsets[0]
    length = (onsets[-1] - shift)  // pip + 1
    grid = [False for _ in range(length)]
    for onset in onsets:
        grid[(onset - shift) // pip] = True
    return grid


