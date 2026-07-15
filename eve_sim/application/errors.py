class CommandValidationError(ValueError):
    """Raised when a typed command cannot be applied atomically."""


class UnsupportedScenarioError(CommandValidationError):
    pass
