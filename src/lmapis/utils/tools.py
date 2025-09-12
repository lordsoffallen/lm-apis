from typing import Optional, Callable, Any, Type, Dict
from pydantic import BaseModel, Field, create_model, ValidationError
from .messages import Tool, ChatCompletionMessageToolCall
from docstring_parser import parse

import inspect
import json


TEXT_EDITOR_TOOL = {
    "type": "text_editor_20250728",
    "name": "str_replace_based_edit_tool"
}

def handle_text_editor_tool(
    tool_call: ChatCompletionMessageToolCall, text: str
) -> Tool | tuple[str, Tool]:
    """
    Handle editor tool commands for string content.

    Args:
        tool_call: The tool call object containing input parameters
        text: The string content to operate on

    Returns:
        Tool class and optionally modified text depending on the tool section
    """
    if tool_call.function.name != TEXT_EDITOR_TOOL["name"]:
        raise ValueError("This function is for str based edit tool")

    input_params = json.loads(tool_call.function.arguments)
    command = input_params.get('command', '')

    if command == 'view':
        return Tool(
            content=text,
            tool_call_id=tool_call.id
        )
    elif command == 'str_replace':
        old_str = input_params.get('old_str', '')
        new_str = input_params.get('new_str', '')

        if not old_str:
            return Tool(
                content="Error: old_str cannot be empty",
                tool_call_id=tool_call.id
            )
        if old_str not in text:
            return Tool(
                content="Error: String not found in content",
                tool_call_id=tool_call.id
            )
        # Check for multiple matches
        match_count = text.count(old_str)
        if match_count > 1:
            return Tool(
                content=f"Error: String appears {match_count} times. "
                        f"Please be more specific.",
                tool_call_id=tool_call.id
            )
        # Perform replacement
        updated_content = text.replace(old_str, new_str, 1)
        return updated_content, Tool(
            content=f"Text replaced successfully",
            tool_call_id=tool_call.id
        )
    elif command == 'insert':
        insert_line = input_params.get('insert_line', 0)
        new_text = input_params.get('new_str', '')

        lines = text.split('\n')

        if insert_line < 0 or insert_line > len(lines):
            return Tool(
                content=f"Error: Invalid line number {insert_line}. "
                        f"Content has {len(lines)} lines.",
                tool_call_id=tool_call.id
            )
        # Insert the new text
        lines.insert(insert_line, new_text)
        updated_content = '\n'.join(lines)

        return updated_content, Tool(
            content=f"Text inserted at line {insert_line}",
            tool_call_id=tool_call.id
        )
    else:
        return Tool(
            content=f"Error: Unknown or unsupported command: {command}",
            tool_call_id=tool_call.id
        )


def _validate_format(format: str) -> str:
    """Validate that the format is supported."""
    supported_formats = ["openai", "anthropic"]
    if format not in supported_formats:
        raise ValueError(f"Unsupported format '{format}'. "
                         f"Supported formats: {supported_formats}")
    return format


class Tools:
    def __init__(self, tools: list[Callable | dict] | dict = None, api_format: str = "openai"):
        """Initialize Tools with optional format specification.
        
        Args:
            tools: List of callable functions or dict tool specifications
            api_format: Output format for tools ("openai" or "anthropic")
        """
        self._tools = {}
        self._format = _validate_format(api_format)
        
        if tools:
            if not isinstance(tools, list):
                tools = [tools]

            for tool in tools:
                self._add_tool(tool)

    @property
    def tools(self):
        return self._tools

    def _add_tool(
        self, func: Callable | dict, param_model: Optional[Type[BaseModel]] = None
    ):
        """Register a tool function with metadata. If no param_model is provided, infer from function signature."""

        if isinstance(func, dict):
            self._tools[func["name"]] = func
        else:
            if param_model:
                tool_spec = self._convert_to_tool_spec(func, param_model)
            else:
                tool_spec, param_model = self.__infer_from_signature(func)

            self._tools[func.__name__] = {
                "function": func,
                "param_model": param_model,
                "spec": tool_spec,
            }

    def format(self, api_format: str = None) -> list:
        """Return tools in the specified format.
        
        Args:
            api_format: Output format ("openai" or "anthropic"). If None, uses instance format.
            
        Returns:
            List of tool specifications in the requested format
        """
        # Use instance format if no format specified
        if api_format is None:
            api_format = self._format
        else:
            # Validate the provided format
            api_format = _validate_format(api_format)
            
        if api_format == "openai":
            result = []
            for tool in self._tools.values():
                if isinstance(tool, dict) and "spec" in tool:
                    # Function-based tool with spec
                    result.append({"type": "function", "function": tool["spec"]})
                elif isinstance(tool, dict):
                    # Dict-based tool (already in spec format)
                    result.append({"type": "function", "function": tool})
                else:
                    # Fallback for unexpected format
                    result.append({"type": "function", "function": tool})
            return result
        elif api_format == "anthropic":
            return [tool for tool in self._tools.values()]
        
        # Default format - return specs only
        result = []
        for tool in self._tools.values():
            if isinstance(tool, dict) and "spec" in tool:
                result.append(tool["spec"])
            elif isinstance(tool, dict):
                result.append(tool)
            else:
                result.append(tool)
        return result

    @staticmethod
    def _convert_to_tool_spec(func: Callable, param_model: Type[BaseModel]) -> dict[str, Any]:
        """Convert the function and its Pydantic model to a unified tool specification."""
        type_mapping = {
            str: "string", int: "integer", float: "number", bool: "boolean"
        }

        properties = {}
        for field_name, field in param_model.model_fields.items():
            field_type = field.annotation

            # Handle enum types
            if hasattr(field_type, "__members__"):  # Check if it's an enum
                enum_values = [
                    member.value if hasattr(member, "value") else member.name
                    for member in field_type
                ]
                properties[field_name] = {
                    "type": "string",
                    "enum": enum_values,
                    "description": field.description or "",
                }
                # Convert enum default value to string if it exists
                if str(field.default) != "PydanticUndefined":
                    properties[field_name]["default"] = (
                        field.default.value
                        if hasattr(field.default, "value")
                        else field.default
                    )
            else:
                properties[field_name] = {
                    "type": type_mapping.get(field_type, str(field_type)),
                    "description": field.description or "",
                }
                # Add default if it exists and isn't PydanticUndefined
                if str(field.default) != "PydanticUndefined":
                    properties[field_name]["default"] = field.default

        return {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [
                    name
                    for name, field in param_model.model_fields.items()
                    if field.is_required and str(field.default) == "PydanticUndefined"
                ],
            },
        }

    @staticmethod
    def __extract_param_descriptions(func: Callable) -> dict[str, str]:
        """Extract parameter descriptions from function docstring.

        Args:
            func: The function to extract parameter descriptions from

        Returns:
            Dictionary mapping parameter names to their descriptions
        """
        docstring = inspect.getdoc(func) or ""
        parsed_docstring = parse(docstring)

        param_descriptions = {}
        for param in parsed_docstring.params:
            param_descriptions[param.arg_name] = param.description or ""

        return param_descriptions

    def __infer_from_signature(self, func: Callable) -> tuple[Dict[str, Any], Type[BaseModel]]:
        """Infer parameters(required and optional) and requirements directly from the function signature."""
        signature = inspect.signature(func)
        fields = {}
        required_fields = []

        # Get function's docstring and parse parameter descriptions
        param_descriptions = self.__extract_param_descriptions(func)
        docstring = inspect.getdoc(func) or ""

        # Parse the docstring to get the main function description
        parsed_docstring = parse(docstring)
        function_description = parsed_docstring.short_description or ""
        if parsed_docstring.long_description:
            function_description += "\n\n" + parsed_docstring.long_description

        for param_name, param in signature.parameters.items():
            # Check if a type annotation is missing
            if param.annotation == inspect._empty:
                raise TypeError(
                    f"Parameter '{param_name}' in function '{func.__name__}' must have a type annotation."
                )

            # Determine field type and optionality
            param_type = param.annotation
            description = param_descriptions.get(param_name, "")

            if param.default == inspect._empty:
                fields[param_name] = (param_type, Field(..., description=description))
                required_fields.append(param_name)
            else:
                fields[param_name] = (
                    param_type,
                    Field(default=param.default, description=description),
                )

        # Dynamically create a Pydantic model based on inferred fields
        param_model = create_model(
            f"{func.__name__.capitalize()}Params", **fields
        )

        # Convert inferred model to a tool spec format
        tool_spec = self._convert_to_tool_spec(func, param_model)

        # Update the tool spec with the parsed function description instead of raw docstring
        tool_spec["description"] = function_description

        return tool_spec, param_model


def execute_tool(
    tools: Tools,
    tool_calls: list[ChatCompletionMessageToolCall] | ChatCompletionMessageToolCall,
    tool_text_input: str = None,
) -> tuple[list[Any], list[Tool]]:
    """Executes registered tools based on the tool calls from the model.

    Args:
        tools: Defined tools for model api call
        tool_calls: List of tool calls from the model
        tool_text_input: Text input for handle text editor function call

    Returns:
        List of tuples containing (result, result_message) for each tool call
    """
    results = []
    messages = []

    if not isinstance(tool_calls, list):
        tool_calls = [tool_calls]

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments
        tool_call_id = tool_call.id

        # Ensure arguments is a dict
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        if tool_name not in tools.tools:
            raise ValueError(f"Tool '{tool_name}' not registered.")

        tool = tools.tools[tool_name]

        tool_func = tool.get("function")
        param_model = tool.get("param_model")

        if tool_func is not None and param_model is not None:
            # Validate and parse the arguments with Pydantic if a model exists
            try:
                validated_args = param_model(**arguments)
                result = tool_func(**validated_args.model_dump())
                results.append(result)
                messages.append(
                    Tool(content=json.dumps(result), tool_call_id=tool_call_id)
                )
            except ValidationError as e:
                raise ValueError(f"Error in tool '{tool_name}' parameters: {e}")
        else:
            # Built in tool call
            if tool == TEXT_EDITOR_TOOL:
                # User passes the tool function to run
                tool_msg = handle_text_editor_tool(tool_call, text=tool_text_input)
                if isinstance(tool_msg, tuple):
                    # unpack the results
                    result, tool_msg = tool_msg
                    results.append(result)
                messages.append(tool_msg)
            else:
                raise NotImplementedError(f"This tool is not implemented yet: {tool}")
    return results, messages
