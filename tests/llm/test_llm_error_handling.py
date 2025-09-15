"""Tests for LLM error handling and custom exceptions."""

import pytest
import os

from unittest.mock import patch

from lmapis.llm import LLM
from lmapis.exceptions import ProhibitedContentError, ContentFilterError, ModelRefusalError
from lmapis.utils.messages import Messages, User


class TestLLMErrorHandling:
    """Test custom error handling in LLM class."""

    def setup_method(self):
        """Set up test fixtures."""
        os.environ["OPENAI_API_KEY"] = "dummykey"
        self.llm = LLM(
            backend="openai",
            model="gpt-4",
            cost={"input": 0.01, "output": 0.03}
        )

    def _create_mock_response(self, finish_reason: str, content: str = "", refusal: str = None):
        """Create a mock ChatCompletion response."""

        return {
            "id": "test_id",
            "model": "gpt-4",
            "object": "chat.completion",
            "choices": [
                {"index": 0,
                 "message": {
                     "role": "assistant",
                     "content": content,
                     "refusal": refusal
                 },
                 "finish_reason": finish_reason
                 }
            ],
            "created": "1234567890"
        }


    def test_prohibited_content_error(self):
        """Test that prohibited content errors are properly raised."""
        mock_response = self._create_mock_response("content_filter: PROHIBITED_CONTENT")
        
        with patch.object(self.llm, '_chat_completion', return_value=mock_response):
            messages = Messages() >> User("Test message")
            
            with pytest.raises(ProhibitedContentError) as exc_info:
                self.llm.chat_completion(messages)
            
            assert "prohibited content policy" in str(exc_info.value).lower()
            assert exc_info.value.response is not None
            assert exc_info.value.response['choices'][0]['finish_reason'] == "content_filter: PROHIBITED_CONTENT"

    def test_content_filter_error(self):
        """Test that general content filter errors are properly raised."""
        mock_response = self._create_mock_response("content_filter: SAFETY")
        
        with patch.object(self.llm, '_chat_completion', return_value=mock_response):
            messages = Messages() >> User("Test message")
            
            with pytest.raises(ContentFilterError) as exc_info:
                self.llm.chat_completion(messages)
            
            assert "content was filtered" in str(exc_info.value).lower()
            assert exc_info.value.filter_reason == "content_filter: SAFETY"
            assert exc_info.value.response is not None

    def test_model_refusal_error(self):
        """Test that model refusal errors are properly raised."""
        mock_response = self._create_mock_response(
            "stop", 
            content="", 
            refusal="I cannot help with that request"
        )
        
        with patch.object(self.llm, '_chat_completion', return_value=mock_response):
            messages = Messages() >> User("Test message")
            
            with pytest.raises(ModelRefusalError) as exc_info:
                self.llm.chat_completion(messages)
            
            assert "model refused to respond" in str(exc_info.value).lower()
            assert exc_info.value.refusal_reason == "I cannot help with that request"
            assert exc_info.value.response is not None

    def test_normal_response_no_error(self):
        """Test that normal responses don't raise errors."""
        mock_response = self._create_mock_response("stop", content="Hello, how can I help?")
        
        with patch.object(self.llm, '_chat_completion', return_value=mock_response):
            messages = Messages() >> User("Test message")
            
            # Should not raise any exception
            response = self.llm.chat_completion(messages)
            assert response["choices"][0]["message"]["content"] == "Hello, how can I help?"

    def test_case_insensitive_content_filter_detection(self):
        """Test that content filter detection is case insensitive."""
        mock_response = self._create_mock_response("CONTENT_FILTER: prohibited_content")
        
        with patch.object(self.llm, '_chat_completion', return_value=mock_response):
            messages = Messages() >> User("Test message")
            
            with pytest.raises(ProhibitedContentError):
                self.llm.chat_completion(messages)

    def test_chat_completion_parse_error_handling(self):
        """Test that chat_completion_parse also handles errors correctly."""
        mock_response = self._create_mock_response("content_filter: PROHIBITED_CONTENT")
        
        with patch.object(self.llm, '_chat_completion_parse', return_value=mock_response):
            messages = Messages() >> User("Test message")
            
            with pytest.raises(ProhibitedContentError):
                self.llm.chat_completion_parse(messages, response_format=str)

    def test_error_logging_integration(self):
        """Test that errors are properly logged when they occur."""
        from lmapis.logging import LoggerConfig, LLMLogger, Console
        
        # Set up logger
        config = LoggerConfig(storage_backends=[Console()])
        logger = LLMLogger(config)
        
        llm_with_logger = LLM(
            backend="openai",
            model="gpt-4",
            cost={"input": 0.01, "output": 0.03},
            logger=logger
        )
        
        mock_response = self._create_mock_response("content_filter: PROHIBITED_CONTENT")
        
        with patch.object(llm_with_logger, '_chat_completion', return_value=mock_response):
            messages = Messages() >> User("Test message")
            
            with pytest.raises(ProhibitedContentError):
                llm_with_logger.chat_completion(messages)
            
            # The error should be logged (we can't easily test the actual logging output
            # without more complex mocking, but we can verify the exception is raised)
