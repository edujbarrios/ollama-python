import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from pydantic.json_schema import JsonSchemaValue
from typing_extensions import Literal

from ollama._client import AsyncClient, Client
from ollama._types import ChatResponse, Message, Options


class Agent:
  """
  A synchronous agent that wraps the Ollama chat API with optional tool calling.

  The agent manages a conversation with an Ollama model.  When tools are
  provided it automatically dispatches tool calls to the registered Python
  functions and feeds results back until a final text response is produced.
  When no tools are provided the agent acts as a plain conversational or
  vision-language model (VLM) wrapper, preserving history across turns and
  accepting images for multimodal evaluation.

  Works with both local Ollama models and Ollama cloud models.

  Tool-calling example::

    from ollama import Agent

    def add(a: int, b: int) -> int:
      '''Add two numbers.

      Args:
        a: First number
        b: Second number

      Returns:
        int: The sum
      '''
      return a + b

    agent = Agent(model='llama3.1', tools=[add])
    response = agent.chat('What is 3 + 4?')
    print(response.message.content)

  Vision / image-evaluation example::

    from ollama import Agent

    agent = Agent(
      model='llava',
      system='You are an expert image reviewer. Evaluate whether the description matches the image.',
    )
    response = agent.chat(
      'Does this description match the image? Description: a red apple on a wooden table.',
      images=['apple.jpg'],
    )
    print(response.message.content)
  """

  def __init__(
    self,
    model: str,
    tools: Optional[Sequence[Callable]] = None,
    *,
    client: Optional[Client] = None,
    system: Optional[str] = None,
    max_iterations: int = 10,
    think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ):
    """
    Create a new Agent.

    Args:
      model: The model to use (e.g. 'llama3.1', 'llava', 'gemma3', or a cloud model).
      tools: An optional sequence of Python callable functions to make available to the
        model.  When omitted (or empty) the agent works as a plain conversational or
        vision-language agent — no tool-calling loop is performed.
        Functions should have Google-style docstrings for best results.
      client: An optional Client instance. If not provided, a default Client is created.
      system: An optional system prompt to guide the agent's behavior.
      max_iterations: Maximum number of tool-calling iterations before returning (default: 10).
        Only relevant when tools are provided.
      think: Enable thinking mode for supported models.
      format: The format of the response.
      options: Model options.
      keep_alive: Keep model alive for the specified duration.
    """
    self.model = model
    self.client = client or Client()
    self.system = system
    self.max_iterations = max_iterations
    self.think = think
    self.format = format
    self.options = options
    self.keep_alive = keep_alive

    self._tools: List[Callable] = list(tools) if tools else []
    self._tool_map: Dict[str, Callable] = {func.__name__: func for func in self._tools}
    self._messages: List[Union[Mapping[str, Any], Message]] = []

    if self.system:
      self._messages.append(Message(role='system', content=self.system))

  @property
  def messages(self) -> List[Union[Mapping[str, Any], Message]]:
    """Return the conversation history."""
    return list(self._messages)

  def reset(self) -> None:
    """Clear conversation history, keeping the system prompt if one was set."""
    self._messages.clear()
    if self.system:
      self._messages.append(Message(role='system', content=self.system))

  def _base_chat_kwargs(self) -> Dict[str, Any]:
    """Return the common kwargs for client.chat calls."""
    return {
      'model': self.model,
      'messages': self._messages,
      'think': self.think,
      'format': self.format,
      'options': self.options,
      'keep_alive': self.keep_alive,
    }

  def chat(
    self,
    message: str,
    *,
    images: Optional[Sequence[Any]] = None,
  ) -> ChatResponse:
    """
    Send a message to the agent and get a response.

    When tools are registered the agent automatically handles tool calls from
    the model, executing the registered functions and feeding results back until
    a final response is produced or max_iterations is reached.

    When no tools are registered (e.g. for vision-language models) the agent
    performs a direct chat and returns the model response immediately, while
    still maintaining the full conversation history across turns.

    Args:
      message: The user message to send.
      images: Optional images for multimodal / vision-language models.

    Returns:
      ChatResponse: The final response from the model.
    """
    user_msg: Dict[str, Any] = {'role': 'user', 'content': message}
    if images:
      user_msg['images'] = images
    self._messages.append(user_msg)

    # Without tools: direct chat without tool-calling loop (VLM / conversational mode)
    if not self._tools:
      response = self.client.chat(**self._base_chat_kwargs())
      self._messages.append(response.message)
      return response

    for _ in range(self.max_iterations):
      response = self.client.chat(
        **self._base_chat_kwargs(),
        tools=self._tools,
      )

      if not response.message.tool_calls:
        self._messages.append(response.message)
        return response

      # Process tool calls
      self._messages.append(response.message)
      for tool_call in response.message.tool_calls:
        func = self._tool_map.get(tool_call.function.name)
        if func:
          try:
            result = func(**tool_call.function.arguments)
          except Exception as e:
            result = f'Error calling {tool_call.function.name}: {e}'
          self._messages.append({
            'role': 'tool',
            'content': str(result),
            'tool_name': tool_call.function.name,
          })
        else:
          self._messages.append({
            'role': 'tool',
            'content': f'Error: function {tool_call.function.name!r} not found',
            'tool_name': tool_call.function.name,
          })

    # Max iterations reached, return last response without tools
    response = self.client.chat(**self._base_chat_kwargs())
    self._messages.append(response.message)
    return response


class AsyncAgent:
  """
  An asynchronous agent that wraps the Ollama chat API with optional tool calling.

  The async agent manages a conversation with an Ollama model.  When tools are
  provided it automatically dispatches tool calls to the registered Python
  functions (including async functions) and feeds results back until a final
  text response is produced.  When no tools are provided the agent acts as a
  plain conversational or vision-language model (VLM) wrapper.

  Works with both local Ollama models and Ollama cloud models.

  Tool-calling example::

    import asyncio
    from ollama import AsyncAgent

    def add(a: int, b: int) -> int:
      '''Add two numbers.

      Args:
        a: First number
        b: Second number

      Returns:
        int: The sum
      '''
      return a + b

    async def main():
      agent = AsyncAgent(model='llama3.1', tools=[add])
      response = await agent.chat('What is 3 + 4?')
      print(response.message.content)

    asyncio.run(main())

  Vision / image-evaluation example::

    import asyncio
    from ollama import AsyncAgent

    async def main():
      agent = AsyncAgent(
        model='llava',
        system='You are an expert image reviewer.',
      )
      response = await agent.chat(
        'Does this description match the image? Description: a red apple.',
        images=['apple.jpg'],
      )
      print(response.message.content)

    asyncio.run(main())
  """

  def __init__(
    self,
    model: str,
    tools: Optional[Sequence[Callable]] = None,
    *,
    client: Optional[AsyncClient] = None,
    system: Optional[str] = None,
    max_iterations: int = 10,
    think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ):
    """
    Create a new AsyncAgent.

    Args:
      model: The model to use (e.g. 'llama3.1', 'llava', 'gemma3', or a cloud model).
      tools: An optional sequence of Python callable functions to make available to the
        model.  When omitted (or empty) the agent works as a plain conversational or
        vision-language agent — no tool-calling loop is performed.
        Functions should have Google-style docstrings for best results.
      client: An optional AsyncClient instance. If not provided, a default AsyncClient is created.
      system: An optional system prompt to guide the agent's behavior.
      max_iterations: Maximum number of tool-calling iterations before returning (default: 10).
        Only relevant when tools are provided.
      think: Enable thinking mode for supported models.
      format: The format of the response.
      options: Model options.
      keep_alive: Keep model alive for the specified duration.
    """
    self.model = model
    self.client = client or AsyncClient()
    self.system = system
    self.max_iterations = max_iterations
    self.think = think
    self.format = format
    self.options = options
    self.keep_alive = keep_alive

    self._tools: List[Callable] = list(tools) if tools else []
    self._tool_map: Dict[str, Callable] = {func.__name__: func for func in self._tools}
    self._messages: List[Union[Mapping[str, Any], Message]] = []

    if self.system:
      self._messages.append(Message(role='system', content=self.system))

  @property
  def messages(self) -> List[Union[Mapping[str, Any], Message]]:
    """Return the conversation history."""
    return list(self._messages)

  def reset(self) -> None:
    """Clear conversation history, keeping the system prompt if one was set."""
    self._messages.clear()
    if self.system:
      self._messages.append(Message(role='system', content=self.system))

  def _base_chat_kwargs(self) -> Dict[str, Any]:
    """Return the common kwargs for client.chat calls."""
    return {
      'model': self.model,
      'messages': self._messages,
      'think': self.think,
      'format': self.format,
      'options': self.options,
      'keep_alive': self.keep_alive,
    }

  async def chat(
    self,
    message: str,
    *,
    images: Optional[Sequence[Any]] = None,
  ) -> ChatResponse:
    """
    Send a message to the agent and get a response.

    When tools are registered the agent automatically handles tool calls from
    the model, executing the registered functions (including async functions)
    and feeding results back until a final response is produced or
    max_iterations is reached.

    When no tools are registered (e.g. for vision-language models) the agent
    performs a direct chat and returns the model response immediately, while
    still maintaining the full conversation history across turns.

    Args:
      message: The user message to send.
      images: Optional images for multimodal / vision-language models.

    Returns:
      ChatResponse: The final response from the model.
    """
    user_msg: Dict[str, Any] = {'role': 'user', 'content': message}
    if images:
      user_msg['images'] = images
    self._messages.append(user_msg)

    # Without tools: direct chat without tool-calling loop (VLM / conversational mode)
    if not self._tools:
      response = await self.client.chat(**self._base_chat_kwargs())
      self._messages.append(response.message)
      return response

    for _ in range(self.max_iterations):
      response = await self.client.chat(
        **self._base_chat_kwargs(),
        tools=self._tools,
      )

      if not response.message.tool_calls:
        self._messages.append(response.message)
        return response

      # Process tool calls
      self._messages.append(response.message)
      for tool_call in response.message.tool_calls:
        func = self._tool_map.get(tool_call.function.name)
        if func:
          try:
            if inspect.iscoroutinefunction(func):
              result = await func(**tool_call.function.arguments)
            else:
              result = func(**tool_call.function.arguments)
          except Exception as e:
            result = f'Error calling {tool_call.function.name}: {e}'
          self._messages.append({
            'role': 'tool',
            'content': str(result),
            'tool_name': tool_call.function.name,
          })
        else:
          self._messages.append({
            'role': 'tool',
            'content': f'Error: function {tool_call.function.name!r} not found',
            'tool_name': tool_call.function.name,
          })

    # Max iterations reached, return last response without tools
    response = await self.client.chat(**self._base_chat_kwargs())
    self._messages.append(response.message)
    return response
