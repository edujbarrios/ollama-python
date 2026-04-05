# Ollama Python Library (Extended with Agent Support)

This is an adapted version of the [official Ollama Python library](https://github.com/ollama/ollama-python) that adds **Agent** and **AsyncAgent** classes for tool-calling and vision-language model (VLM) workflows.

The library provides the easiest way to integrate Python 3.8+ projects with [Ollama](https://github.com/ollama/ollama), now with built-in agent capabilities.

## Prerequisites

- [Ollama](https://ollama.com/download) should be installed and running
- Pull a model to use with the library: `ollama pull <model>` e.g. `ollama pull gemma3`
  - See [Ollama.com](https://ollama.com/search) for more information on the models available.

## Install

```sh
pip install ollama
```

## Development Install

To install this adapted library from source for testing and development:

```sh
git clone https://github.com/edujbarrios/ollama-python.git
cd ollama-python
pip install -e .
```

This installs the package in editable mode so that any local changes are immediately available without reinstalling.

> **Note:** This will replace the official `ollama` package. To revert, run `pip install ollama` to restore the upstream version.

## Usage

```python
from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='gemma3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)
```

See [_types.py](ollama/_types.py) for more information on the response types.

## Streaming responses

Response streaming can be enabled by setting `stream=True`.

```python
from ollama import chat

stream = chat(
    model='gemma3',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
```

## Cloud Models

Run larger models by offloading to Ollama’s cloud while keeping your local workflow.

- Supported models: `deepseek-v3.1:671b-cloud`, `gpt-oss:20b-cloud`, `gpt-oss:120b-cloud`, `kimi-k2:1t-cloud`, `qwen3-coder:480b-cloud`, `kimi-k2-thinking` See [Ollama Models - Cloud](https://ollama.com/search?c=cloud) for more information

### Run via local Ollama

1) Sign in (one-time):

```
ollama signin
```

2) Pull a cloud model:

```
ollama pull gpt-oss:120b-cloud
```

3) Make a request:

```python
from ollama import Client

client = Client()

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b-cloud', messages=messages, stream=True):
  print(part.message.content, end='', flush=True)
```

### Cloud API (ollama.com)

Access cloud models directly by pointing the client at `https://ollama.com`.

1) Create an API key from [ollama.com](https://ollama.com/settings/keys) , then set:

```
export OLLAMA_API_KEY=your_api_key
```

2) (Optional) List models available via the API:

```
curl https://ollama.com/api/tags
```

3) Generate a response via the cloud API:

```python
import os
from ollama import Client

client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
)

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
  print(part.message.content, end='', flush=True)
```

## Custom client
A custom client can be created by instantiating `Client` or `AsyncClient` from `ollama`.

All extra keyword arguments are passed into the [`httpx.Client`](https://www.python-httpx.org/api/#client).

```python
from ollama import Client
client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)
response = client.chat(model='gemma3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
```

## Async client

The `AsyncClient` class is used to make asynchronous requests. It can be configured with the same fields as the `Client` class.

```python
import asyncio
from ollama import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  response = await AsyncClient().chat(model='gemma3', messages=[message])

asyncio.run(chat())
```

Setting `stream=True` modifies functions to return a Python asynchronous generator:

```python
import asyncio
from ollama import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  async for part in await AsyncClient().chat(model='gemma3', messages=[message], stream=True):
    print(part['message']['content'], end='', flush=True)

asyncio.run(chat())
```

## Agent

The `Agent` class provides a high-level wrapper around the Ollama chat API with automatic tool-calling support. The agent manages conversation history and handles the tool-calling loop automatically.

### Agent with Tools

Register Python functions as tools and the agent will automatically call them when the model requests it:

```python
from ollama import Agent

def add_two_numbers(a: int, b: int) -> int:
  """
  Add two numbers together.

  Args:
    a (int): The first number
    b (int): The second number

  Returns:
    int: The sum of the two numbers
  """
  return a + b

def subtract_two_numbers(a: int, b: int) -> int:
  """
  Subtract two numbers.

  Args:
    a (int): The first number
    b (int): The second number

  Returns:
    int: The difference of the two numbers
  """
  return a - b

agent = Agent(
  model='llama3.1',
  tools=[add_two_numbers, subtract_two_numbers],
  system='You are a helpful math assistant.',
)

response = agent.chat('What is three plus one?')
print(response.message.content)

# The agent maintains conversation history across turns
response = agent.chat('Now subtract 2 from that result.')
print(response.message.content)
```

### Vision Agent (VLM)

When no tools are provided, the agent works as a vision-language model wrapper, supporting image inputs:

```python
from ollama import Agent

agent = Agent(
  model='llava',
  system='You are an expert image reviewer. Evaluate whether the description matches the image.',
)

response = agent.chat(
  'Does this description match the image? Description: a red apple on a wooden table.',
  images=['path/to/image.jpg'],
)
print(response.message.content)

# Follow-up questions retain full conversation context
followup = agent.chat('What specific details in the image led to your verdict?')
print(followup.message.content)
```

### Async Agent

The `AsyncAgent` class provides the same capabilities as `Agent` but for asynchronous workflows:

```python
import asyncio
from ollama import AsyncAgent

def add_two_numbers(a: int, b: int) -> int:
  """
  Add two numbers together.

  Args:
    a (int): The first number
    b (int): The second number

  Returns:
    int: The sum of the two numbers
  """
  return a + b

async def main():
  agent = AsyncAgent(
    model='llama3.1',
    tools=[add_two_numbers],
    system='You are a helpful math assistant.',
  )

  response = await agent.chat('What is three plus one?')
  print(response.message.content)

asyncio.run(main())
```

### Multi-Agent Vision Conversations

Multiple `Agent` instances can collaborate on the same image, each with a different role. The output of one agent is passed as input to the next, creating a pipeline:

```python
from ollama import Agent

image_path = 'path/to/image.jpg'

# Agent 1 — describe the image
describer = Agent(
  model='gemma3',
  system='You are an expert image describer. Provide a rich, detailed description.',
)

description = describer.chat(
  'Describe this image in detail.',
  images=[image_path],
).message.content

# Agent 2 — critique the description
critic = Agent(
  model='gemma3',
  system='You are a meticulous image-description critic. Point out inaccuracies or missing details.',
)

critique = critic.chat(
  f'Here is a description of the attached image:\n\n"{description}"\n\n'
  'What is accurate? What is missing or wrong?',
  images=[image_path],
).message.content

# Agent 3 — produce a final summary
summarizer = Agent(
  model='gemma3',
  system='You are a skilled summarizer. Write a concise, accurate summary incorporating the feedback.',
)

summary = summarizer.chat(
  f'Original description:\n"{description}"\n\n'
  f'Critic feedback:\n"{critique}"\n\n'
  'Write a final, concise summary of the image.',
  images=[image_path],
).message.content

print(summary)
```

See [examples/multi-agent-vision.py](examples/multi-agent-vision.py) and [examples/async-multi-agent-vision.py](examples/async-multi-agent-vision.py) for complete working examples.

See also [examples/agent.py](examples/agent.py), [examples/async-agent.py](examples/async-agent.py), and [examples/vision-agent.py](examples/vision-agent.py) for single-agent examples.

## API

The Ollama Python library's API is designed around the [Ollama REST API](https://github.com/ollama/ollama/blob/main/docs/api.md)

### Chat

```python
ollama.chat(model='gemma3', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
```

### Generate

```python
ollama.generate(model='gemma3', prompt='Why is the sky blue?')
```

### List

```python
ollama.list()
```

### Show

```python
ollama.show('gemma3')
```

### Create

```python
ollama.create(model='example', from_='gemma3', system="You are Mario from Super Mario Bros.")
```

### Copy

```python
ollama.copy('gemma3', 'user/gemma3')
```

### Delete

```python
ollama.delete('gemma3')
```

### Pull

```python
ollama.pull('gemma3')
```

### Push

```python
ollama.push('user/gemma3')
```

### Embed

```python
ollama.embed(model='gemma3', input='The sky is blue because of rayleigh scattering')
```

### Embed (batch)

```python
ollama.embed(model='gemma3', input=['The sky is blue because of rayleigh scattering', 'Grass is green because of chlorophyll'])
```

### Ps

```python
ollama.ps()
```

## Errors

Errors are raised if requests return an error status or if an error is detected while streaming.

```python
model = 'does-not-yet-exist'

try:
  ollama.chat(model)
except ollama.ResponseError as e:
  print('Error:', e.error)
  if e.status_code == 404:
    ollama.pull(model)
```
