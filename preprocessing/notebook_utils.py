import os
import warnings
from IPython import get_ipython

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

def is_notebook() -> bool:
    """
    Returns True if running in a Jupyter notebook, False otherwise.
    """
    try:
        # Get the shell object from IPython
        shell = get_ipython().__class__.__name__
        # Check if we're in a notebook-like environment
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':  # Terminal IPython
            return False
        else:
            return False
    except NameError:  # If get_ipython is not defined (standard Python interpreter)
        return False
