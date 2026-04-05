"""
Async multi-agent conversation with vision-language models.

Same pipeline as multi-agent-vision.py but using AsyncAgent for
non-blocking I/O.  This is useful when integrating into async web
frameworks or when orchestrating many agents concurrently.

Pipeline:
  1. **Describer** — generates a detailed description of the image.
  2. **Critic** — reviews the description for accuracy and completeness.
  3. **Summarizer** — produces a concise final summary incorporating feedback.

Usage:
  python examples/async-multi-agent-vision.py path/to/image.jpg

Requirements:
  A vision-capable model must be pulled first, e.g.:
    ollama pull gemma3
    ollama pull llava
"""

import asyncio
import sys

from ollama import AsyncAgent


async def main() -> None:
  if len(sys.argv) < 2:
    print('Usage: python async-multi-agent-vision.py <image_path>')
    sys.exit(1)

  image_path = sys.argv[1]
  model = 'gemma3'  # any vision-capable model

  # --- Agent 1: Describer ---------------------------------------------------
  describer = AsyncAgent(
    model=model,
    system=(
      'You are an expert image describer. '
      'Given an image, provide a rich and detailed description covering the '
      'main subjects, background, colours, lighting, mood, and any text visible.'
    ),
  )

  print('=== Describer Agent ===')
  description_response = await describer.chat(
    'Please describe this image in detail.',
    images=[image_path],
  )
  description = description_response.message.content
  print(description)
  print()

  # --- Agent 2: Critic -------------------------------------------------------
  critic = AsyncAgent(
    model=model,
    system=(
      'You are a meticulous image-description critic. '
      'You will receive an image together with a written description. '
      'Point out any inaccuracies, missing details, or exaggerations. '
      'Be constructive and specific.'
    ),
  )

  print('=== Critic Agent ===')
  critique_response = await critic.chat(
    f'Here is a description of the attached image:\n\n"{description}"\n\n'
    'Please critique the description. What is accurate? What is missing or wrong?',
    images=[image_path],
  )
  critique = critique_response.message.content
  print(critique)
  print()

  # --- Agent 3: Summarizer ---------------------------------------------------
  summarizer = AsyncAgent(
    model=model,
    system=(
      'You are a skilled summarizer. '
      'Given an original image description and a critic\'s review, '
      'produce a concise, accurate final summary of the image that '
      'incorporates the feedback.'
    ),
  )

  print('=== Summarizer Agent ===')
  summary_response = await summarizer.chat(
    f'Original description:\n"{description}"\n\n'
    f'Critic feedback:\n"{critique}"\n\n'
    'Please write a final, concise, and accurate summary of the image.',
    images=[image_path],
  )
  print(summary_response.message.content)
  print()

  # --- Follow-up: ask the summarizer for a one-line caption ------------------
  print('=== Follow-up (caption) ===')
  caption_response = await summarizer.chat(
    'Now condense that summary into a single-sentence caption suitable for social media.'
  )
  print(caption_response.message.content)


if __name__ == '__main__':
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print('\nGoodbye!')
