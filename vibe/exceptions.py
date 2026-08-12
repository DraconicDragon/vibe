class SessionError(Exception):
    """Raised when session setup or inference fails."""


class TransformError(SessionError):
    """Raised specifically when a result transform fails inside the pipeline."""


class TransformRequirementError(TransformError):
    """Raised by a transform when a prerequisite (like model-provided data) is missing."""


class InferenceCancelled(SessionError):
    """Raised when an in-flight inference run is cancelled by user request."""


class RegistryError(Exception):
    """Raised when a plugin lookup fails."""


class LoaderError(Exception):
    """Raised when file resolution or validation fails."""


class HFDownloadError(Exception):
    """Raised when a HuggingFace download/cached lookup cannot be satisfied."""
