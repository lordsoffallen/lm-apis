"""Custom exceptions for LLM API interactions."""


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class ProhibitedContentError(LLMError):
    """Raised when content is filtered due to prohibited content policies."""
    
    def __init__(self, message: str = "Content was filtered due to prohibited content", response: dict = None):
        super().__init__(message)
        self.response = response


class ContentFilterError(LLMError):
    """Raised when content is filtered by the model's safety systems."""
    
    def __init__(self, message: str = "Content was filtered", filter_reason: str = None, response: dict = None):
        super().__init__(message)
        self.filter_reason = filter_reason
        self.response = response


class ModelRefusalError(LLMError):
    """Raised when the model refuses to respond to a request."""
    
    def __init__(self, message: str = "Model refused to respond", refusal_reason: str = None, response: dict = None):
        super().__init__(message)
        self.refusal_reason = refusal_reason
        self.response = response