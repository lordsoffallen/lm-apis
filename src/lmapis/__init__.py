try:
    from ._version import version as __version__
except ImportError:
    # If the package is not installed with setuptools_scm
    __version__ = "0.0.0"