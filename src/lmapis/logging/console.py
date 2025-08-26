"""
Console storage backend for flexible LLM logging system.

This module provides a console-based storage backend that outputs log entries
to stdout using structlog for clean formatting and optional colors.
"""

import sys
from typing import Any, Dict, Optional, TextIO

import structlog

from .storage import StorageBackend


class Console(StorageBackend):
    """
    Console storage backend that outputs log entries to stdout using structlog.
    
    Supports two formatting styles:
    - 'pretty': Multi-line, human-readable format with colors (if terminal supports it)
    - 'compact': Single-line JSON format suitable for log aggregation
    
    Features:
    - Clean formatting via structlog
    - Automatic color detection and formatting
    - Graceful error handling that doesn't interrupt LLM operations
    - Configurable output stream (stdout by default)
    """
    
    def __init__(
        self, 
        format_style: str = "pretty",
        output_stream: Optional[TextIO] = None
    ):
        """
        Initialize the Console storage backend.
        
        Args:
            format_style: Either 'pretty' or 'compact' formatting
            output_stream: Output stream to write to (defaults to sys.stdout)
            
        Raises:
            ValueError: If format_style is not 'pretty' or 'compact'
        """
        if format_style not in ('pretty', 'compact'):
            raise ValueError("format_style must be either 'pretty' or 'compact'")
        
        self.format_style = format_style
        self.output_stream = output_stream or sys.stdout
        
        # Configure structlog logger based on format style
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up structlog logger with appropriate processors and formatting."""
        if self.format_style == "pretty":
            # Pretty format with colors and human-readable layout
            processors = [
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.ConsoleRenderer(colors=self._supports_color())
            ]
        else:  # compact
            # Compact JSON format
            processors = [
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.JSONRenderer()
            ]
        
        # Configure structlog for this instance
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
            logger_factory=structlog.WriteLoggerFactory(file=self.output_stream),
            cache_logger_on_first_use=False,  # Don't cache to avoid conflicts between instances
        )
        
        # Create logger instance
        self.logger = structlog.get_logger("llm_logger")
    
    def _supports_color(self) -> bool:
        """
        Detect if the terminal supports ANSI color codes.
        
        Returns:
            True if colors are supported, False otherwise
        """
        # Check if output is a TTY and not redirected
        if not hasattr(self.output_stream, 'isatty') or not self.output_stream.isatty():
            return False
        
        # Check common environment variables that indicate color support
        import os
        term = os.environ.get('TERM', '').lower()
        colorterm = os.environ.get('COLORTERM', '').lower()
        
        # Common terminals that support color
        color_terms = ['xterm', 'xterm-color', 'xterm-256color', 'screen', 'linux', 'cygwin']
        
        return (
            any(color_term in term for color_term in color_terms) or
            colorterm in ['truecolor', '24bit']
        )
    
    def save(self, data: Dict[str, Any]) -> None:
        """
        Save a log entry to the console output using structlog.
        
        Args:
            data: Dictionary representation of a log entry
        """
        try:
            # Extract key fields for structured logging
            event_data = {
                'request_id': data.get('request_id'),
                'model': data.get('model'),
                'backend': data.get('backend'),
                'cost': data.get('cost'),
                'tokens_prompt': data.get('tokens_prompt'),
                'tokens_completion': data.get('tokens_completion'),
                'duration_ms': data.get('duration_ms'),
                'finish_reason': data.get('finish_reason'),
            }
            
            # Add request/response content if available (truncated for readability)
            if data.get('messages') and isinstance(data['messages'], list) and data['messages']:
                last_msg = data['messages'][-1]
                if isinstance(last_msg, dict) and 'content' in last_msg:
                    content = str(last_msg['content'])
                    if len(content) > 100:
                        content = content[:97] + "..."
                    event_data['request'] = content
            
            if data.get('response_content'):
                response = str(data['response_content'])
                if len(response) > 150:
                    response = response[:147] + "..."
                event_data['response'] = response
            
            # Remove None values to keep output clean
            event_data = {k: v for k, v in event_data.items() if v is not None}
            
            # Log based on whether there's an error
            if data.get('error'):
                self.logger.error(
                    "LLM request failed",
                    error=data['error'],
                    **event_data
                )
            else:
                self.logger.info(
                    "LLM request completed",
                    **event_data
                )
                
        except Exception as e:
            # Catch any unexpected errors in logging
            try:
                fallback_msg = f"Console logging error: {e}. Request ID: {data.get('request_id', 'unknown')}"
                sys.stderr.write(fallback_msg + '\n')
                sys.stderr.flush()
            except (IOError, OSError):
                # Silently continue if we can't even write to stderr
                pass
    
    def close(self) -> None:
        """
        Clean up console resources.
        
        For console output, this mainly ensures any buffered output is flushed.
        """
        try:
            if hasattr(self.output_stream, 'flush'):
                self.output_stream.flush()
        except (IOError, OSError):
            # Ignore errors during cleanup
            pass