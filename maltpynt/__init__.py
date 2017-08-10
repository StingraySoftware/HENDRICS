# Found at https://stackoverflow.com/questions/24322927/python-how-to-alias-
# module-name-rename-with-preserving-backward-compatibility
import sys

# make sure bar is in sys.modules
import hendrics
# link this module to bar
sys.modules[__name__] = sys.modules['hendrics']
