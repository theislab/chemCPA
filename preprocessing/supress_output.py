import os
import warnings

# Define a context manager to suppress output
class suppress_output:
    def __enter__(self):
        self._stdout = os.dup(1)
        self._stderr = os.dup(2)
        self._null = os.open(os.devnull, os.O_RDWR)
        os.dup2(self._null, 1)
        os.dup2(self._null, 2)
        return self

    def __exit__(self, *args):
        # First restore the original file descriptors
        os.dup2(self._stdout, 1)
        os.dup2(self._stderr, 2)
        # Then close all our saved descriptors
        os.close(self._stdout)
        os.close(self._stderr)
        os.close(self._null)
