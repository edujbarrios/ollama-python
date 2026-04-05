import json

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

from ollama._agent import Agent, AsyncAgent
from ollama._client import AsyncClient, Client

pytestmark = pytest.mark.anyio

# Minimal 1x1 red pixel PNG encoded as base64, used in VLM image tests
_TINY_PNG_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI6QAAAABJRU5ErkJggg=='


@pytest.fixture
def anyio_backend():
  return 'asyncio'


def _add(a: int, b: int) -> int:
  """
  Add two numbers.

  Args:
    a (int): First number
    b (int): Second number

  Returns:
    int: The sum
  """
  return int(a) + int(b)


def _subtract(a: int, b: int) -> int:
  """
  Subtract two numbers.

  Args:
    a (int): First number
    b (int): Second number

  Returns:
    int: The difference
  """
  return int(a) - int(b)


def _make_chat_handler(responses):
  """Create an HTTP handler that returns different responses on successive calls."""
  call_count = [0]

  def handler(request: Request) -> Response:
    idx = min(call_count[0], len(responses) - 1)
    call_count[0] += 1
    return Response(
      json.dumps(responses[idx]),
      content_type='application/json',
    )

  return handler


def test_agent_no_tool_calls(httpserver: HTTPServer):
  """Test agent with a response that has no tool calls."""
  httpserver.expect_request('/api/chat', method='POST').respond_with_json({
    'model': 'dummy',
    'message': {
      'role': 'assistant',
      'content': 'The answer is 4.',
    },
  })

  client = Client(httpserver.url_for('/'))
  agent = Agent(model='dummy', tools=[_add], client=client)
  response = agent.chat('What is 2 + 2?')

  assert response.message.content == 'The answer is 4.'
  assert response.message.role == 'assistant'
  assert len(agent.messages) == 2  # user + assistant


def test_agent_with_tool_call(httpserver: HTTPServer):
  """Test agent handling a tool call and returning the final response."""
  responses = [
    # First response: model requests a tool call
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
          {
            'function': {
              'name': '_add',
              'arguments': {'a': 3, 'b': 1},
            }
          }
        ],
      },
    },
    # Second response: final answer after tool result
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': 'The result of 3 + 1 is 4.',
      },
    },
  ]

  httpserver.expect_request('/api/chat', method='POST').respond_with_handler(
    _make_chat_handler(responses)
  )

  client = Client(httpserver.url_for('/'))
  agent = Agent(model='dummy', tools=[_add, _subtract], client=client)
  response = agent.chat('What is 3 + 1?')

  assert response.message.content == 'The result of 3 + 1 is 4.'
  # Messages: user, assistant (tool_call), tool result, assistant (final)
  assert len(agent.messages) == 4
  assert agent.messages[0] == {'role': 'user', 'content': 'What is 3 + 1?'}


def test_agent_with_system_prompt(httpserver: HTTPServer):
  """Test agent with a system prompt."""
  httpserver.expect_request('/api/chat', method='POST').respond_with_json({
    'model': 'dummy',
    'message': {
      'role': 'assistant',
      'content': 'Hello! I am your math assistant.',
    },
  })

  client = Client(httpserver.url_for('/'))
  agent = Agent(
    model='dummy',
    tools=[_add],
    client=client,
    system='You are a helpful math assistant.',
  )
  response = agent.chat('Hello!')

  assert response.message.content == 'Hello! I am your math assistant.'
  # Messages: system, user, assistant
  assert len(agent.messages) == 3
  assert agent.messages[0].role == 'system'
  assert agent.messages[0].content == 'You are a helpful math assistant.'


def test_agent_reset(httpserver: HTTPServer):
  """Test agent reset clears history but keeps system prompt."""
  httpserver.expect_request('/api/chat', method='POST').respond_with_json({
    'model': 'dummy',
    'message': {
      'role': 'assistant',
      'content': 'Hello!',
    },
  })

  client = Client(httpserver.url_for('/'))
  agent = Agent(
    model='dummy',
    tools=[_add],
    client=client,
    system='System prompt.',
  )

  agent.chat('Hello!')
  assert len(agent.messages) == 3  # system + user + assistant

  agent.reset()
  assert len(agent.messages) == 1  # Only system prompt
  assert agent.messages[0].role == 'system'


def test_agent_reset_no_system(httpserver: HTTPServer):
  """Test agent reset with no system prompt."""
  httpserver.expect_request('/api/chat', method='POST').respond_with_json({
    'model': 'dummy',
    'message': {
      'role': 'assistant',
      'content': 'Hello!',
    },
  })

  client = Client(httpserver.url_for('/'))
  agent = Agent(model='dummy', tools=[_add], client=client)

  agent.chat('Hello!')
  assert len(agent.messages) == 2

  agent.reset()
  assert len(agent.messages) == 0


def test_agent_unknown_tool(httpserver: HTTPServer):
  """Test agent handles calls to unknown functions gracefully."""
  responses = [
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
          {
            'function': {
              'name': 'unknown_func',
              'arguments': {'x': 1},
            }
          }
        ],
      },
    },
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': 'Sorry, I could not find that function.',
      },
    },
  ]

  httpserver.expect_request('/api/chat', method='POST').respond_with_handler(
    _make_chat_handler(responses)
  )

  client = Client(httpserver.url_for('/'))
  agent = Agent(model='dummy', tools=[_add], client=client)
  response = agent.chat('Call unknown function')

  assert response.message.content == 'Sorry, I could not find that function.'
  # Check the error message was sent as tool result
  tool_msg = agent.messages[2]
  assert tool_msg['role'] == 'tool'
  assert 'not found' in tool_msg['content']


def test_agent_tool_exception(httpserver: HTTPServer):
  """Test agent handles tool function exceptions gracefully."""

  def failing_func(x: int) -> int:
    """
    A function that always fails.

    Args:
      x (int): A number

    Returns:
      int: Never returns
    """
    raise ValueError('something went wrong')

  responses = [
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
          {
            'function': {
              'name': 'failing_func',
              'arguments': {'x': 1},
            }
          }
        ],
      },
    },
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': 'The function encountered an error.',
      },
    },
  ]

  httpserver.expect_request('/api/chat', method='POST').respond_with_handler(
    _make_chat_handler(responses)
  )

  client = Client(httpserver.url_for('/'))
  agent = Agent(model='dummy', tools=[failing_func], client=client)
  response = agent.chat('Call the failing function')

  assert response.message.content == 'The function encountered an error.'
  tool_msg = agent.messages[2]
  assert tool_msg['role'] == 'tool'
  assert 'Error calling failing_func' in tool_msg['content']
  assert 'something went wrong' in tool_msg['content']


def test_agent_max_iterations(httpserver: HTTPServer):
  """Test agent respects max_iterations limit."""
  # Always return a tool call to force max iterations
  tool_call_response = {
    'model': 'dummy',
    'message': {
      'role': 'assistant',
      'content': '',
      'tool_calls': [
        {
          'function': {
            'name': '_add',
            'arguments': {'a': 1, 'b': 1},
          }
        }
      ],
    },
  }

  final_response = {
    'model': 'dummy',
    'message': {
      'role': 'assistant',
      'content': 'Final answer after max iterations.',
    },
  }

  responses = [tool_call_response, tool_call_response, tool_call_response, final_response]

  httpserver.expect_request('/api/chat', method='POST').respond_with_handler(
    _make_chat_handler(responses)
  )

  client = Client(httpserver.url_for('/'))
  agent = Agent(model='dummy', tools=[_add], client=client, max_iterations=3)
  response = agent.chat('Keep adding')

  assert response.message.content == 'Final answer after max iterations.'


def test_agent_conversation_history(httpserver: HTTPServer):
  """Test agent maintains conversation history across multiple calls."""
  httpserver.expect_request('/api/chat', method='POST').respond_with_json({
    'model': 'dummy',
    'message': {
      'role': 'assistant',
      'content': 'Response.',
    },
  })

  client = Client(httpserver.url_for('/'))
  agent = Agent(model='dummy', tools=[_add], client=client)

  agent.chat('First message')
  agent.chat('Second message')

  assert len(agent.messages) == 4  # user1, assistant1, user2, assistant2
  assert agent.messages[2] == {'role': 'user', 'content': 'Second message'}


def test_agent_multiple_tool_calls(httpserver: HTTPServer):
  """Test agent handles multiple tool calls in a single response."""
  responses = [
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
          {
            'function': {
              'name': '_add',
              'arguments': {'a': 3, 'b': 1},
            }
          },
          {
            'function': {
              'name': '_subtract',
              'arguments': {'a': 10, 'b': 5},
            }
          },
        ],
      },
    },
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': '3 + 1 = 4 and 10 - 5 = 5.',
      },
    },
  ]

  httpserver.expect_request('/api/chat', method='POST').respond_with_handler(
    _make_chat_handler(responses)
  )

  client = Client(httpserver.url_for('/'))
  agent = Agent(model='dummy', tools=[_add, _subtract], client=client)
  response = agent.chat('What is 3 + 1 and 10 - 5?')

  assert response.message.content == '3 + 1 = 4 and 10 - 5 = 5.'
  # Messages: user, assistant (tool_calls), tool result 1, tool result 2, assistant (final)
  assert len(agent.messages) == 5


# --- Async Agent Tests ---


async def test_async_agent_no_tool_calls(httpserver: HTTPServer):
  """Test async agent with a response that has no tool calls."""
  httpserver.expect_request('/api/chat', method='POST').respond_with_json({
    'model': 'dummy',
    'message': {
      'role': 'assistant',
      'content': 'The answer is 4.',
    },
  })

  client = AsyncClient(httpserver.url_for('/'))
  agent = AsyncAgent(model='dummy', tools=[_add], client=client)
  response = await agent.chat('What is 2 + 2?')

  assert response.message.content == 'The answer is 4.'
  assert len(agent.messages) == 2


async def test_async_agent_with_tool_call(httpserver: HTTPServer):
  """Test async agent handling a tool call."""
  responses = [
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
          {
            'function': {
              'name': '_add',
              'arguments': {'a': 3, 'b': 1},
            }
          }
        ],
      },
    },
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': 'The result is 4.',
      },
    },
  ]

  httpserver.expect_request('/api/chat', method='POST').respond_with_handler(
    _make_chat_handler(responses)
  )

  client = AsyncClient(httpserver.url_for('/'))
  agent = AsyncAgent(model='dummy', tools=[_add, _subtract], client=client)
  response = await agent.chat('What is 3 + 1?')

  assert response.message.content == 'The result is 4.'
  assert len(agent.messages) == 4


async def test_async_agent_with_async_tool(httpserver: HTTPServer):
  """Test async agent with an async tool function."""

  async def async_add(a: int, b: int) -> int:
    """
    Add two numbers asynchronously.

    Args:
      a (int): First number
      b (int): Second number

    Returns:
      int: The sum
    """
    return int(a) + int(b)

  responses = [
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
          {
            'function': {
              'name': 'async_add',
              'arguments': {'a': 5, 'b': 3},
            }
          }
        ],
      },
    },
    {
      'model': 'dummy',
      'message': {
        'role': 'assistant',
        'content': 'The result is 8.',
      },
    },
  ]

  httpserver.expect_request('/api/chat', method='POST').respond_with_handler(
    _make_chat_handler(responses)
  )

  client = AsyncClient(httpserver.url_for('/'))
  agent = AsyncAgent(model='dummy', tools=[async_add], client=client)
  response = await agent.chat('What is 5 + 3?')

  assert response.message.content == 'The result is 8.'
  # Check the tool result was correctly recorded
  tool_msg = agent.messages[2]
  assert tool_msg['role'] == 'tool'
  assert tool_msg['content'] == '8'


async def test_async_agent_reset(httpserver: HTTPServer):
  """Test async agent reset."""
  httpserver.expect_request('/api/chat', method='POST').respond_with_json({
    'model': 'dummy',
    'message': {
      'role': 'assistant',
      'content': 'Hello!',
    },
  })

  client = AsyncClient(httpserver.url_for('/'))
  agent = AsyncAgent(
    model='dummy',
    tools=[_add],
    client=client,
    system='System prompt.',
  )

  await agent.chat('Hello!')
  assert len(agent.messages) == 3

  agent.reset()
  assert len(agent.messages) == 1
  assert agent.messages[0].role == 'system'


# --- No-tools (VLM / conversational) Agent Tests ---


def test_agent_no_tools(httpserver: HTTPServer):
  """Test agent created without tools performs a direct chat."""
  httpserver.expect_request('/api/chat', method='POST').respond_with_json({
    'model': 'dummy',
    'message': {
      'role': 'assistant',
      'content': 'I can see a red apple on a wooden table.',
    },
  })

  client = Client(httpserver.url_for('/'))
  agent = Agent(model='dummy', client=client)
  response = agent.chat('Describe this image.')

  assert response.message.content == 'I can see a red apple on a wooden table.'
  assert len(agent.messages) == 2  # user + assistant


def test_agent_no_tools_with_images(httpserver: HTTPServer):
  """Test no-tools agent accepts images (VLM mode)."""
  body_seen = []

  def handler(request: Request) -> Response:
    body_seen.append(request.get_json())
    return Response(
      json.dumps({
        'model': 'dummy',
        'message': {
          'role': 'assistant',
          'content': 'Correct, the image shows a red apple.',
        },
      }),
      content_type='application/json',
    )

  httpserver.expect_request('/api/chat', method='POST').respond_with_handler(handler)

  client = Client(httpserver.url_for('/'))
  agent = Agent(model='dummy', system='You are an image reviewer.', client=client)
  response = agent.chat('Does this match: a red apple?', images=[_TINY_PNG_B64])

  assert response.message.content == 'Correct, the image shows a red apple.'
  # Images must have been forwarded in the user message
  user_msg = body_seen[0]['messages'][1]
  assert user_msg['images'] is not None
  # No tools should be registered — client sends an empty tools list
  assert body_seen[0].get('tools', []) == []


def test_agent_no_tools_conversation_history(httpserver: HTTPServer):
  """Test no-tools agent preserves conversation history across turns."""
  httpserver.expect_request('/api/chat', method='POST').respond_with_json({
    'model': 'dummy',
    'message': {
      'role': 'assistant',
      'content': 'Reply.',
    },
  })

  client = Client(httpserver.url_for('/'))
  agent = Agent(model='dummy', client=client)

  agent.chat('Turn one.')
  agent.chat('Turn two.')

  assert len(agent.messages) == 4  # user1, assistant1, user2, assistant2
  assert agent.messages[0] == {'role': 'user', 'content': 'Turn one.'}
  assert agent.messages[2] == {'role': 'user', 'content': 'Turn two.'}


async def test_async_agent_no_tools(httpserver: HTTPServer):
  """Test async agent created without tools performs a direct chat."""
  httpserver.expect_request('/api/chat', method='POST').respond_with_json({
    'model': 'dummy',
    'message': {
      'role': 'assistant',
      'content': 'Vision response.',
    },
  })

  client = AsyncClient(httpserver.url_for('/'))
  agent = AsyncAgent(model='dummy', client=client)
  response = await agent.chat('Describe this image.')

  assert response.message.content == 'Vision response.'
  assert len(agent.messages) == 2


async def test_async_agent_no_tools_with_images(httpserver: HTTPServer):
  """Test async no-tools agent accepts images (VLM mode)."""
  body_seen = []

  def handler(request: Request) -> Response:
    body_seen.append(request.get_json())
    return Response(
      json.dumps({
        'model': 'dummy',
        'message': {
          'role': 'assistant',
          'content': 'Correct.',
        },
      }),
      content_type='application/json',
    )

  httpserver.expect_request('/api/chat', method='POST').respond_with_handler(handler)

  client = AsyncClient(httpserver.url_for('/'))
  agent = AsyncAgent(model='dummy', system='You are an image reviewer.', client=client)
  response = await agent.chat('Does this match: a cat?', images=[_TINY_PNG_B64])

  assert response.message.content == 'Correct.'
  user_msg = body_seen[0]['messages'][1]
  assert user_msg['images'] is not None
  assert body_seen[0].get('tools', []) == []
