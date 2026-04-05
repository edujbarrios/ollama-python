import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from pydantic.json_schema import JsonSchemaValue
from typing_extensions import Literal

from ollama._client import AsyncClient, Client
from ollama._types import ChatResponse, Message, Options


class Agent:
  """
  A synchronous agent that wraps the Ollama chat API with automatic tool calling.

  The agent manages a conversation with an Ollama model, automatically dispatching
  tool calls to registered Python functions and feeding results back to the model
  until a final text response is produced.

  Works with both local Ollama models and Ollama cloud models.

  Example::

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
  """

  def __init__(
    self,
    model: str,
    tools: Sequence[Callable],
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
      model: The model to use (e.g. 'llama3.1', 'gemma3', or a cloud model).
      tools: A sequence of Python callable functions to make available to the model.
        Functions should have Google-style docstrings for best results.
      client: An optional Client instance. If not provided, a default Client is created.
      system: An optional system prompt to guide the agent's behavior.
      max_iterations: Maximum number of tool-calling iterations before returning (default: 10).
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

    self._tools: List[Callable] = list(tools)
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

  def chat(
    self,
    message: str,
    *,
    images: Optional[Sequence[Any]] = None,
  ) -> ChatResponse:
    """
    Send a message to the agent and get a response.

    The agent will automatically handle tool calls from the model, executing the
    registered functions and feeding results back until a final response is produced
    or max_iterations is reached.

    Args:
      message: The user message to send.
      images: Optional images for multimodal models.

    Returns:
      ChatResponse: The final response from the model.
    """
    user_msg: Dict[str, Any] = {'role': 'user', 'content': message}
    if images:
      user_msg['images'] = images
    self._messages.append(user_msg)

    for _ in range(self.max_iterations):
      response = self.client.chat(
        model=self.model,
        messages=self._messages,
        tools=self._tools,
        think=self.think,
        format=self.format,
        options=self.options,
        keep_alive=self.keep_alive,
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
    response = self.client.chat(
      model=self.model,
      messages=self._messages,
      think=self.think,
      format=self.format,
      options=self.options,
      keep_alive=self.keep_alive,
    )
    self._messages.append(response.message)
    return response


class AsyncAgent:
  """
  An asynchronous agent that wraps the Ollama chat API with automatic tool calling.

  The async agent manages a conversation with an Ollama model, automatically
  dispatching tool calls to registered Python functions (including async functions)
  and feeding results back to the model until a final text response is produced.

  Works with both local Ollama models and Ollama cloud models.

  Example::

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
  """

  def __init__(
    self,
    model: str,
    tools: Sequence[Callable],
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
      model: The model to use (e.g. 'llama3.1', 'gemma3', or a cloud model).
      tools: A sequence of Python callable functions to make available to the model.
        Functions should have Google-style docstrings for best results.
      client: An optional AsyncClient instance. If not provided, a default AsyncClient is created.
      system: An optional system prompt to guide the agent's behavior.
      max_iterations: Maximum number of tool-calling iterations before returning (default: 10).
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

    self._tools: List[Callable] = list(tools)
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

  async def chat(
    self,
    message: str,
    *,
    images: Optional[Sequence[Any]] = None,
  ) -> ChatResponse:
    """
    Send a message to the agent and get a response.

    The agent will automatically handle tool calls from the model, executing the
    registered functions (including async functions) and feeding results back until
    a final response is produced or max_iterations is reached.

    Args:
      message: The user message to send.
      images: Optional images for multimodal models.

    Returns:
      ChatResponse: The final response from the model.
    """
    user_msg: Dict[str, Any] = {'role': 'user', 'content': message}
    if images:
      user_msg['images'] = images
    self._messages.append(user_msg)

    for _ in range(self.max_iterations):
      response = await self.client.chat(
        model=self.model,
        messages=self._messages,
        tools=self._tools,
        think=self.think,
        format=self.format,
        options=self.options,
        keep_alive=self.keep_alive,
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
    response = await self.client.chat(
      model=self.model,
      messages=self._messages,
      think=self.think,
      format=self.format,
      options=self.options,
      keep_alive=self.keep_alive,
    )
    self._messages.append(response.message)
    return response
