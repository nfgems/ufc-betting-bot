"""Atomic exclusive publication without a transient shared hard link."""
import ctypes
import os
import sys


def publish_new(temporary, destination):
    """Rename complete bytes exactly once; refuse an existing destination."""
    if sys.platform.startswith('linux'):
        library = ctypes.CDLL(None, use_errno=True)
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(temporary), -100, os.fsencode(destination), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
    elif os.name == 'nt':
        # Windows rename refuses an existing destination, including a file.
        os.rename(temporary, destination)
    else:
        raise OSError('Exclusive atomic publication is unavailable on this platform')
