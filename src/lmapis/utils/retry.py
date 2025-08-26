from requests.exceptions import ConnectionError, Timeout, HTTPError

import re

# Define which HTTP status codes should NOT be retried
NON_RETRYABLE_STATUS_CODES = {
    400,  # Bad Request - client error, won't change on retry
    401,  # Unauthorized - authentication issue
    403,  # Forbidden - authorization issue
    404,  # Not Found - endpoint doesn't exist
    405,  # Method Not Allowed - wrong HTTP method
    406,  # Not Acceptable - content negotiation failure
    409,  # Conflict - resource conflict
    410,  # Gone - resource permanently removed
    413,  # Payload Too Large - request too big
    414,  # URI Too Long - URL too long
    415,  # Unsupported Media Type - wrong content type
    422,  # Unprocessable Entity - validation error
    451,  # Unavailable For Legal Reasons - legal restriction
}


def should_retry_exception(exception):
    """
    Determine if an exception should trigger a retry.
    Returns True if we should retry, False otherwise.
    """
    # Always retry on connection/network issues
    if isinstance(exception, (ConnectionError, Timeout)):
        return True

    # Handle standard requests HTTPError
    if isinstance(exception, HTTPError):
        if hasattr(exception, 'response') and exception.response is not None:
            status_code = exception.response.status_code
            if status_code in NON_RETRYABLE_STATUS_CODES:
                return False
            if status_code >= 500 or status_code in {408, 429}:
                return True

    # Extract status code from various possible locations
    status_code = None

    # Method 1: Direct status_code attribute
    if hasattr(exception, 'status_code'):
        status_code = exception.status_code

    # Method 2: code attribute (common in API clients)
    elif hasattr(exception, 'code'):
        status_code = exception.code

    # Method 3: Parse from exception message
    elif hasattr(exception, 'args') and exception.args:
        # Look for "Error code: XXX" pattern
        error_msg = str(exception.args[0]) if exception.args else str(exception)
        code_match = re.search(
            r'(?:error code|code):\s*(\d+)', error_msg, re.IGNORECASE
        )
        if code_match:
            status_code = int(code_match.group(1))

    # Method 4: Parse JSON error responses
    if not status_code:
        try:
            error_str = str(exception)
            # Try to extract code from JSON-like error messages
            if "'code': " in error_str:
                code_match = re.search(r"'code':\s*(\d+)", error_str)
                if code_match:
                    status_code = int(code_match.group(1))
        except:
            pass

    # If we found a status code, check if it's retryable
    if status_code:
        if status_code in NON_RETRYABLE_STATUS_CODES:
            return False
        if status_code >= 500 or status_code in {408, 429}:
            return True

    # Fallback: Check exception message for common non-retryable patterns
    exception_str = str(exception).lower()
    non_retryable_patterns = [
        'not found', 'not_found', '404',
        'bad request', '400',
        'unauthorized', '401',
        'forbidden', '403',
        'method not allowed', '405',
        'not acceptable', '406',
        'conflict', '409',
        'gone', '410',
        'payload too large', '413',
        'uri too long', '414',
        'unsupported media type', '415',
        'unprocessable entity', '422',
        'invalid', 'malformed'
    ]

    if any(pattern in exception_str for pattern in non_retryable_patterns):
        return False

    # Default to not retry for unknown exceptions
    return False