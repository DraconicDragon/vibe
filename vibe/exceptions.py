from typing import Any


class SessionError(Exception):
    """Raised when session setup or inference fails."""


class InferenceCancelled(SessionError):
    """Raised when an in-flight inference run is cancelled by user request."""

    def __init__(
        self,
        message: str = "Inference cancelled by user request.",
        partial_result: Any = None,
    ) -> None:
        super().__init__(message)
        self.partial_result = partial_result


class RegistryError(Exception):
    """Raised when a plugin lookup fails."""


class LoaderError(Exception):
    """Raised when file resolution or validation fails."""


class HFDownloadError(Exception):
    """Raised when a HuggingFace download/cached lookup cannot be satisfied."""


class PluginContractError(Exception):
    """Raised at load time if a plugin fails to satisfy its declared capability contracts."""


class SessionCapabilityError(Exception):
    """Raised if a user attempts to access a capability view (e.g. .tagger) on an incompatible session."""
