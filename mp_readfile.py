from __future__ import division, print_function
import sys
from mp_io import mp_get_file_type, is_string
import collections

if __name__ == '__main__':
    fname = sys.argv[1]
    ftype, contents = mp_get_file_type(fname)
    print ('-----------------------------')
    print ('This file contains:', end='\n\n')
    for k in sorted(contents.keys()):
        val = contents[k]
        if isinstance(val, collections.Iterable) and not is_string(val):
            if len(val) < 4:
                val = repr(list(val[:4]))
            else:
                val = repr(list(val[:4])).replace(']', '') + '...]'
        print ((k + ':').ljust(15), val, end='\n\n')

    print ('-----------------------------')
