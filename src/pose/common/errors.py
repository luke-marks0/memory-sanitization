class PoseError(Exception):
    """Base repository exception."""


class ConfigurationError(PoseError):
    """Raised when repository or profile configuration is invalid."""


class ProtocolError(PoseError):
    """Raised when a protocol message or result artifact is invalid."""


class ResourceFailure(PoseError):
    """Raised when a required local or hardware resource is unavailable."""


class UnsupportedPhaseError(PoseError):
    """Raised when a requested operation belongs to a later implementation phase."""

