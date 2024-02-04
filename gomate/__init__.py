try:
    from .version import version as __version__
except ImportError:
    __version__ = "unknown version"

from . import modules