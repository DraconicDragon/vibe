class SessionError(Exception):
    """Raised when session setup or inference fails."""


class ProcessorError(SessionError):
    """Raised specifically when a result processor fails inside the pipeline."""


class InferenceCancelled(SessionError):
    """Raised when an in-flight inference run is cancelled by user request."""
