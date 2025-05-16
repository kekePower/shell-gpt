import json
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Union, cast

from ..cache import Cache
from ..config import cfg
from ..function import get_function
from ..printer import MarkdownPrinter, Printer, TextPrinter
from ..role import DefaultRoles, SystemRole

# Type aliases
CompletionGenerator = Generator[Any, None, None]
CompletionFunc = Callable[..., CompletionGenerator]

# Initialize completion function placeholder
completion: CompletionFunc = lambda *args, **kwargs: (yield from [])

# Get configuration values
base_url = cfg.get("API_BASE_URL")
use_litellm = cfg.get("USE_LITELLM") == "true"
use_ollama = cfg.get("USE_OLLAMA") == "true"

# Common request parameters
request_params = {
    "timeout": int(cfg.get("REQUEST_TIMEOUT")),
    "api_key": cfg.get("OPENAI_API_KEY"),
}

# Initialize the appropriate client
if use_ollama:
    # For Ollama, we'll use the OpenAI client but point it to the Ollama server
    ollama_base_url = cfg.get("OLLAMA_BASE_URL")
    request_params.update({
        "base_url": ollama_base_url,
        "api_key": "ollama",  # Ollama doesn't require an API key
    })
    from openai import OpenAI
    
    client = OpenAI(**request_params)
    completion = client.chat.completions.create
    additional_kwargs = {
        "model": cfg.get("OLLAMA_MODEL"),
        "temperature": float(cfg.get("OLLAMA_TEMPERATURE")),
        "top_p": float(cfg.get("OLLAMA_TOP_P")),
    }
elif use_litellm:
    import litellm  # type: ignore

    completion = litellm.completion
    litellm.suppress_debug_info = True
    # LiteLLM doesn't use the same parameter names
    additional_kwargs = {k: v for k, v in request_params.items() if k != "api_key"}
else:
    # Standard OpenAI client
    if base_url != "default":
        request_params["base_url"] = base_url
    
    from openai import OpenAI
    from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageToolCall
    
    client = OpenAI(**request_params)
    completion = client.chat.completions.create
    additional_kwargs = {}


class Handler:
    cache = Cache(int(cfg.get("CACHE_LENGTH")), Path(cfg.get("CACHE_PATH")))

    def __init__(self, role: SystemRole, markdown: bool) -> None:
        self.role = role

        api_base_url = cfg.get("API_BASE_URL")
        self.base_url = None if api_base_url == "default" else api_base_url
        self.timeout = int(cfg.get("REQUEST_TIMEOUT"))

        self.markdown = "APPLY MARKDOWN" in self.role.role and markdown
        self.code_theme, self.color = cfg.get("CODE_THEME"), cfg.get("DEFAULT_COLOR")

    @property
    def printer(self) -> Printer:
        return (
            MarkdownPrinter(self.code_theme)
            if self.markdown
            else TextPrinter(self.color)
        )

    def make_messages(self, prompt: str) -> List[Dict[str, str]]:
        raise NotImplementedError

    def handle_function_call(
        self,
        messages: List[dict[str, Any]],
        name: str,
        arguments: str,
    ) -> Generator[str, None, None]:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "function_call": {"name": name, "arguments": arguments},
            }
        )

        if messages and messages[-1]["role"] == "assistant":
            yield "\n"

        dict_args = json.loads(arguments)
        joined_args = ", ".join(f'{k}="{v}"' for k, v in dict_args.items())
        yield f"> @FunctionCall `{name}({joined_args})` \n\n"

        result = get_function(name)(**dict_args)
        if cfg.get("SHOW_FUNCTIONS_OUTPUT") == "true":
            yield f"```text\n{result}\n```\n"
        messages.append({"role": "function", "content": result, "name": name})

    @cache
    def get_completion(
        self,
        model: str,
        temperature: float,
        top_p: float,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, str]]],
    ) -> Generator[str, None, None]:
        name = arguments = ""
        is_shell_role = self.role.name == DefaultRoles.SHELL.value
        is_code_role = self.role.name == DefaultRoles.CODE.value
        is_dsc_shell_role = self.role.name == DefaultRoles.DESCRIBE_SHELL.value
        use_ollama = cfg.get("USE_OLLAMA") == "true"
        
        # Disable functions for certain roles or when using Ollama
        if is_shell_role or is_code_role or is_dsc_shell_role or use_ollama:
            functions = None

        # Prepare the request parameters
        request_params = {
            "model": model if not use_ollama else cfg.get("OLLAMA_MODEL"),
            "messages": messages,
            "stream": True,
            **additional_kwargs,
        }
        
        # Only add these parameters if they're not already in additional_kwargs
        if not use_ollama:
            request_params.update({
                "temperature": temperature,
                "top_p": top_p,
            })

        # Handle tools/functions if provided (not supported by Ollama)
        if functions and not use_ollama:
            request_params["tools"] = functions
            request_params["tool_choice"] = "auto"

        # Make the API call
        try:
            response = completion(**request_params)
        except Exception as e:
            raise RuntimeError(f"Failed to get completion from the API: {str(e)}")

        try:
            name = ""
            arguments = ""
            
            for chunk in response:
                if not hasattr(chunk, 'choices') or not chunk.choices:
                    continue
                    
                choice = chunk.choices[0]
                delta = choice.delta
                
                # Handle tool calls if present
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if hasattr(tool_call, 'function') and tool_call.function:
                            if hasattr(tool_call.function, 'name') and tool_call.function.name:
                                name = tool_call.function.name
                            if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                arguments += tool_call.function.arguments
                
                # Handle finish reason
                if hasattr(choice, 'finish_reason') and choice.finish_reason == "tool_calls":
                    if name and arguments:
                        yield from self.handle_function_call(messages, name, arguments)
                        # Continue with a new completion to handle the function response
                        yield from self.get_completion(
                            model=model,
                            temperature=temperature,
                            top_p=top_p,
                            messages=messages,
                            functions=functions,
                            caching=False,
                        )
                    return
                
                # Yield the content if available
                if hasattr(delta, 'content') and delta.content is not None:
                    yield delta.content
                    
        except KeyboardInterrupt:
            if hasattr(response, 'close'):
                response.close()

    def handle(
        self,
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        caching: bool,
        functions: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        disable_stream = cfg.get("DISABLE_STREAMING") == "true"
        messages = self.make_messages(prompt.strip())
        generator = self.get_completion(
            model=model,
            temperature=temperature,
            top_p=top_p,
            messages=messages,
            functions=functions,
            caching=caching,
            **kwargs,
        )
        return self.printer(generator, not disable_stream)
