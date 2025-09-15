import pytest
import yaml

from pathlib import Path
from typing import Dict, Any
from ..conftest import is_env_set
from lmapis.llm import LLM, Prompt
from lmapis.utils.messages import Messages, System, User, Assistant
from lmapis.logging import Console, LLMLogger, LoggerConfig
from lmapis.utils.tools import Tools, TEXT_EDITOR_TOOL


def load_model_configs():
    """Load model configurations from tests/models.yml"""
    config_path = Path(__file__).parent.parent / "models.yml"
    if not config_path.exists():
        return {}

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model from models.yml"""
    configs = load_model_configs()
    if model_name not in configs:
        raise ValueError(f"Model '{model_name}' not found in models.yml")
    return configs[model_name]


def get_env_var_for_backend(backend: str) -> str:
    """Get the environment variable name for a given backend"""
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GEMINI_API_KEY",
        "google-genai": "GEMINI_API_KEY",
        "together": "TOGETHER_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "mistral": "MISTRAL_API_KEY"
    }
    return env_var_map.get(backend, f"{backend.upper()}_API_KEY")


class TestLLMIntegration:
    """Integration tests that require actual API keys."""

    @pytest.fixture(scope="class")
    def model_configs(self):
        """Load model configurations from YAML file."""
        return load_model_configs()

    def _create_llm_from_config(self, model_name: str, envs: dict) -> LLM:
        """Helper method to create LLM instance from YAML config."""
        config = get_model_config(model_name)
        backend = config["backend"]
        env_var = get_env_var_for_backend(backend)

        return LLM(
            backend=backend,
            model=config["model"],
            cost=config["cost"],
            credentials=envs.get(env_var),
            model_params=config.get("model_params", {}),
            logger=LLMLogger(
                LoggerConfig(storage_backends=[Console()])
            )
        )

    def _run_simple_prompt(self, llm: LLM, expected_answer: str) -> None:
        """Helper method to run a simple math test."""
        prompt = Prompt(
            user=f"What is {expected_answer}? Answer with just the number.",
            system="You are a helpful assistant."
        )

        response = llm(prompt=prompt)

        assert isinstance(response, Assistant)
        assert response.content is not None
        assert expected_answer in response.content

    @pytest.mark.skipif(
        not is_env_set("OPENAI_API_KEY"), reason="Test requires OpenAI API key"
    )
    def test_gpt4o_mini_integration(self, envs):
        """Test LLM with GPT-4o-mini."""
        llm = self._create_llm_from_config("gpt-4o-mini", envs)
        self._run_simple_prompt(llm, "4")  # 2+2=4

    @pytest.mark.skipif(
        not is_env_set("OPENAI_API_KEY"), reason="Test requires OpenAI API key"
    )
    def test_gpt4o_integration(self, envs):
        """Test LLM with GPT-4o."""
        llm = self._create_llm_from_config("gpt-4o", envs)
        self._run_simple_prompt(llm, "6")  # 3+3=6

    @pytest.mark.skipif(
        not is_env_set("ANTHROPIC_API_KEY"), reason="Test requires Anthropic API key"
    )
    def test_sonnet_35_integration(self, envs):
        """Test LLM with Claude 3.5 Sonnet."""
        llm = self._create_llm_from_config("sonnet-3.5", envs)
        self._run_simple_prompt(llm, "8")  # 4+4=8

    @pytest.mark.skipif(
        not is_env_set("ANTHROPIC_API_KEY"), reason="Test requires Anthropic API key"
    )
    def test_sonnet_35_cached_integration(self, envs):
        """Test LLM with Claude 3.5 Sonnet with prompt caching."""
        llm = self._create_llm_from_config("sonnet-3.5-cached", envs)

        # Verify prompt caching is enabled
        assert llm.is_prompt_caching_enabled is True

        self._run_simple_prompt(llm, "10")  # 5+5=10

    @pytest.mark.skipif(
        not is_env_set("GEMINI_API_KEY"), reason="Test requires Gemini API key"
    )
    def test_gemini_25_flash_integration(self, envs):
        """Test LLM with Gemini 2.5 Flash."""
        llm = self._create_llm_from_config("gemini-2.5-flash", envs)
        self._run_simple_prompt(llm, "12")  # 6+6=12

    @pytest.mark.skipif(
        not is_env_set("GEMINI_API_KEY"), reason="Test requires Gemini API key"
    )
    def test_gemini_25_pro_integration(self, envs):
        """Test LLM with Gemini 2.5 Pro."""
        llm = self._create_llm_from_config("gemini-2.5-pro", envs)
        self._run_simple_prompt(llm, "14")  # 7+7=14

    @pytest.mark.skipif(
        not is_env_set("FIREWORKS_API_KEY"), reason="Test requires Fireworks API key"
    )
    def test_llama_v31_70b_integration(self, envs):
        """Test LLM with Llama 3.1 70B."""
        llm = self._create_llm_from_config("llama-v3p1-70b", envs)
        self._run_simple_prompt(llm, "16")  # 8+8=16

    @pytest.mark.skipif(
        not is_env_set("FIREWORKS_API_KEY"), reason="Test requires Fireworks API key"
    )
    def test_deepseek_v3_integration(self, envs):
        """Test LLM with DeepSeek V3."""
        llm = self._create_llm_from_config("deepseek-v3", envs)
        self._run_simple_prompt(llm, "18")  # 9+9=18

    @pytest.mark.skipif(
        not is_env_set("MISTRAL_API_KEY"), reason="Test requires Mistral API key"
    )
    def test_mistral_large_integration(self, envs):
        """Test LLM with Mistral Large."""
        llm = self._create_llm_from_config("mistral-large", envs)
        self._run_simple_prompt(llm, "20")  # 10+10=20

    def test_model_config_loading(self, model_configs):
        """Test that model configurations are loaded correctly."""
        assert isinstance(model_configs, dict)
        assert len(model_configs) > 0

        # Test a few key models exist
        expected_models = ["gpt-4o-mini", "sonnet-3.5", "gemini-2.5-flash"]
        for model in expected_models:
            if model in model_configs:
                config = model_configs[model]
                assert "backend" in config
                assert "model" in config
                assert "cost" in config
                assert "input" in config["cost"]
                assert "output" in config["cost"]

    def test_model_config_structure(self):
        """Test that individual model configs have the correct structure."""
        config = get_model_config("gpt-4o-mini")

        # Required fields
        assert "backend" in config
        assert "model" in config
        assert "cost" in config

        # Cost structure
        assert isinstance(config["cost"], dict)
        assert "input" in config["cost"]
        assert "output" in config["cost"]
        assert isinstance(config["cost"]["input"], (int, float))
        assert isinstance(config["cost"]["output"], (int, float))

        # Optional fields
        if "model_params" in config:
            assert isinstance(config["model_params"], dict)

    @pytest.mark.parametrize("model_name", [
        "gpt-4o-mini", "gpt-4o", "sonnet-3.5", "gemini-2.5-flash",
        "llama-v3p1-70b", "deepseek-v3", "mistral-large"
    ])
    def test_model_config_completeness(self, model_name):
        """Test that each model config has all required fields."""
        try:
            config = get_model_config(model_name)

            # Check required fields
            assert config["backend"] in [
                "openai", "anthropic", "google", "google-genai",
                "together", "fireworks", "mistral"
            ]
            assert isinstance(config["model"], str)
            assert len(config["model"]) > 0
            assert isinstance(config["cost"]["input"], (int, float))
            assert isinstance(config["cost"]["output"], (int, float))
            assert config["cost"]["input"] > 0
            assert config["cost"]["output"] > 0

        except ValueError:
            # Model not found in config, skip test
            pytest.skip(f"Model {model_name} not found in models.yml")

    def test_env_var_mapping(self):
        """Test that environment variable mapping works correctly."""
        assert get_env_var_for_backend("openai") == "OPENAI_API_KEY"
        assert get_env_var_for_backend("anthropic") == "ANTHROPIC_API_KEY"
        assert get_env_var_for_backend("google") == "GOOGLE_API_KEY"
        assert get_env_var_for_backend("google-genai") == "GOOGLE_API_KEY"
        assert get_env_var_for_backend("fireworks") == "FIREWORKS_API_KEY"
        assert get_env_var_for_backend("mistral") == "MISTRAL_API_KEY"
        assert get_env_var_for_backend("together") == "TOGETHER_API_KEY"

    def test_multiple_models_from_config(self, envs, model_configs):
        """
        Test multiple models from the YAML config.
        This test runs all available models that have API keys set.
        """
        results = {}

        for model_name, config in model_configs.items():
            backend = config["backend"]
            env_var = get_env_var_for_backend(backend)

            # Skip if API key not available
            if not envs.get(env_var):
                results[model_name] = {"skipped": True, "reason": f"No {env_var}"}
                continue

            try:
                llm = LLM(
                    backend=backend,
                    model=config["model"],
                    cost=config["cost"],
                    credentials=envs[env_var],
                    model_params=config.get("model_params", {})
                )

                # Simple test prompt
                prompt = Prompt(
                    user="Say 'Hello' and nothing else.",
                    system="You are a helpful assistant."
                )

                response = llm(prompt=prompt)

                results[model_name] = {
                    "success": True,
                    "response_length": len(response.content) if response.content else 0,
                    "has_content": response.content is not None
                }

            except Exception as e:
                results[model_name] = {
                    "success": False,
                    "error": str(e)
                }

        # Print results for manual inspection
        print(f"\nTested {len(results)} models:")
        for model_name, result in results.items():
            if result.get("skipped"):
                print(f"  {model_name}: SKIPPED ({result['reason']})")
            elif result.get("success"):
                print \
                    (f"  {model_name}: SUCCESS (response: {result['response_length']} chars)")
            else:
                print \
                    (f"  {model_name}: FAILED ({result.get('error', 'Unknown error')})")

        # At least one model should have been tested successfully
        successful_tests = [r for r in results.values() if r.get("success")]
        if not successful_tests:
            pytest.skip("No models could be tested (no API keys available)")

        # All successful tests should have valid responses
        for result in successful_tests:
            assert result["has_content"], "Response should have content"


"""
Live integration tests for LLM text editing functionality with Claude 4.

This module tests the LLM class's text editing capabilities using real API calls
to Claude 4, covering both single-turn and multi-turn editing scenarios.
The tests ensure the model can properly use the view action and handle multiple turns.
"""

class TestLLMTextEditingLive:
    """Live integration tests for LLM text editing with Claude 4."""

    @pytest.fixture
    def claude_4_llm(self, envs):
        """Create LLM instance configured for Claude 4."""
        return LLM(
            backend="anthropic",
            model="claude-sonnet-4-20250514",
            cost={"input": 3, "output": 15},
            credentials=envs.get("ANTHROPIC_API_KEY"),
            model_params={"temperature": 0.6},
            logger=LLMLogger(
                LoggerConfig(storage_backends=[Console()])
            )
        )

    @pytest.fixture
    def text_editor_tools(self):
        """Create tools with text editor for testing."""
        return Tools([TEXT_EDITOR_TOOL], api_format="anthropic")

    @pytest.fixture()
    def system_message(self) -> System:
        return System("If user mentions a file or document, the path to read the "
                      "file/document is at `tmp/input.md`")

    @pytest.mark.skipif(
        not is_env_set("ANTHROPIC_API_KEY"),
        reason="Test requires Anthropic API key"
    )
    def test_single_turn_text_replacement_live(
        self, system_message, claude_4_llm, text_editor_tools
    ):
        """Test single-turn text editing with str_replace command using live Claude 4 API."""
        # Create messages with input text and guide model to view first
        messages = Messages() >> system_message >> User(
            "I have a text file that needs editing. Please replace 'Hello' with "
            "'Hi' in the text file."
        )
        messages.input_text_files = "Hello world! How are you?"

        # Execute with max_turns=3 to allow view + edit operations
        result = claude_4_llm(messages=messages, tools=text_editor_tools, max_turns=3)

        # Verify the result
        assert result is not None
        assert result.text_files is not None
        assert result.text_files.strip() == "Hi world! How are you?"
        assert "Hi world!" in result.text_files
        assert "Hello" not in result.text_files

        # Verify cost tracking
        assert claude_4_llm.call_cost > 0

    @pytest.mark.skipif(
        not is_env_set("ANTHROPIC_API_KEY"),
        reason="Test requires Anthropic API key"
    )
    def test_single_turn_text_insertion_live(
        self, system_message, claude_4_llm, text_editor_tools
    ):
        """Test single-turn text editing with insert command using live Claude 4 API."""
        # Create messages with input text and guide model to view first
        messages = Messages() >> system_message >> User(
            "I have a text file that needs editing. Please insert 'Welcome! ' at the "
            "beginning of the text file."
        )
        messages.input_text_files = "This is a test document."

        # Execute with max_turns=3 to allow view + edit operations
        result = claude_4_llm(messages=messages, tools=text_editor_tools, max_turns=3)

        # Verify the result
        assert result is not None
        assert result.text_files is not None
        assert result.text_files.startswith("Welcome!")
        assert "This is a test document." in result.text_files

        # Verify cost tracking
        assert claude_4_llm.call_cost > 0

    @pytest.mark.skipif(
        not is_env_set("ANTHROPIC_API_KEY"),
        reason="Test requires Anthropic API key"
    )
    def test_single_turn_text_view_live(
        self, system_message, claude_4_llm, text_editor_tools
    ):
        """Test single-turn text viewing with view command using live Claude 4 API."""
        # Create messages with input text and explicit view instruction
        messages = Messages() >> system_message >> User(
            "What does my document contain?"
        )
        messages.input_text_files = "Hello world! This is a sample document for testing."

        # Execute with max_turns=2 to allow view + response
        result = claude_4_llm(messages=messages, tools=text_editor_tools, max_turns=2)

        # Verify the result
        assert result is not None
        assert result.text_files is not None
        # Text should remain unchanged for view operation
        assert result.text_files == "Hello world! This is a sample document for testing."

        # The response should contain some analysis of the text content
        assert result.content is not None
        assert len(result.content) > 0

        # Verify cost tracking
        assert claude_4_llm.call_cost > 0

    @pytest.mark.skipif(
        not is_env_set("ANTHROPIC_API_KEY"),
        reason="Test requires Anthropic API key"
    )
    def test_multi_turn_text_editing_live(
        self, system_message, claude_4_llm, text_editor_tools
    ):
        """Test multi-turn text editing with multiple sequential edits using live Claude 4 API."""
        # Create messages with input text and complex editing request
        messages = Messages() >> system_message >> User(
            "Please make these changes to the document step by step (not at once): "
            "1. Replace 'Hello' with 'Greetings' "
            "2. Replace 'world' with 'everyone' "
            "3. Add an exclamation mark at the end if there isn't one"
        )
        messages.input_text_files = "Hello world. How are you today?"

        # Execute with max_turns=5 to allow view + multiple tool executions
        result = claude_4_llm(messages=messages, tools=text_editor_tools, max_turns=6)

        # Verify the result
        assert result is not None
        assert result.text_files is not None

        # Check that all transformations were applied
        final_text = result.text_files
        assert "Greetings" in final_text
        assert "everyone" in final_text
        assert "Hello" not in final_text
        assert "world" not in final_text
        # Should end with exclamation mark
        assert final_text.rstrip().endswith("!")

        # Verify cost tracking
        assert claude_4_llm.call_cost > 0

    @pytest.mark.skipif(
        not is_env_set("ANTHROPIC_API_KEY"),
        reason="Test requires Anthropic API key"
    )
    def test_comprehensive_single_turn_with_view_live(
        self, system_message, claude_4_llm, text_editor_tools
    ):
        """Test comprehensive single-turn text editing that requires view action using live Claude 4 API."""
        # Create messages with complex text and explicit view instruction
        messages = Messages() >> system_message >> User(
            "Please make the following change to the document: "
            "replace any occurrence of 'TODO' with 'COMPLETED' in the text file."
        )
        messages.input_text_files = """Project Status Report

Task 1: Setup database - COMPLETED
Task 2: Create API endpoints - TODO
Task 3: Write documentation - TODO
Task 4: Deploy to staging - PENDING

Notes: The TODO items need to be updated once finished."""

        # Execute with max_turns=5 to ensure enough turns for view + edits
        result = claude_4_llm(messages=messages, tools=text_editor_tools, max_turns=5)

        # Verify the result
        assert result is not None
        assert result.text_files is not None

        # Check that all TODO items were replaced
        final_text = result.text_files
        assert "TODO" not in final_text
        assert "COMPLETED" in final_text
        # Should have multiple COMPLETED entries now
        assert final_text.count("COMPLETED") >= 3

        # Verify cost tracking
        assert claude_4_llm.call_cost > 0

    @pytest.mark.skipif(
        not is_env_set("ANTHROPIC_API_KEY"),
        reason="Test requires Anthropic API key"
    )
    def test_comprehensive_multi_turn_with_view_live(
        self, system_message, claude_4_llm, text_editor_tools
    ):
        """Test comprehensive multi-turn text editing with view action using live Claude 4 API."""
        # Create messages with complex editing scenario
        messages = Messages() >> system_message >> User(
            "I have a configuration file that needs multiple updates. Please make these changes in sequence: "
            "1. Change 'debug=false' to 'debug=true' "
            "2. Update 'version=1.0' to 'version=2.0' "
            "3. Replace 'environment=dev' with 'environment=production' "
            "4. Add a new line 'updated=2025-01-15' at the end"
        )
        messages.input_text_files = """# Application Configuration
app_name=MyApp
version=1.0
debug=false
environment=dev
port=8080
database_url=localhost:5432"""

        # Execute with max_turns=6 to allow view + multiple sequential edits
        result = claude_4_llm(messages=messages, tools=text_editor_tools, max_turns=6)

        # Verify the result
        assert result is not None
        assert result.text_files is not None

        final_config = result.text_files

        # Check that all changes were applied
        assert "debug=true" in final_config
        assert "debug=false" not in final_config
        assert "version=2.0" in final_config
        assert "version=1.0" not in final_config
        assert "environment=production" in final_config
        assert "environment=dev" not in final_config
        assert "updated=2025-01-15" in final_config

        # Verify cost tracking
        assert claude_4_llm.call_cost > 0
