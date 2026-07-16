"""Custom AI Brain exception types."""


class AIBrainError(Exception):
    """Base exception for AI Brain module errors."""


class ConfigurationError(AIBrainError):
    """Raised when required configuration is missing or invalid."""


class AnalysisPipelineError(AIBrainError):
    """Raised when orchestration fails during analysis."""
