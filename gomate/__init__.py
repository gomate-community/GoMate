try:
    from .version import __version__
except ImportError:
    __version__ = "unknown version"

from . import modules