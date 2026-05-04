"""
custom exceptions for babbage pipeline.
"""


class BabbageError(Exception):
    """base exception for babbage pipeline."""
    pass


class PipelineIOError(BabbageError):
    """file I/O error in pipeline."""
    pass


class ValidationError(BabbageError):
    """input validation error."""
    pass
