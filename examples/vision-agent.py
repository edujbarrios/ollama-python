"""
Vision-language agent example.

Demonstrates using Agent without tools to evaluate whether a natural-language
description correctly matches the content of an image.  The agent keeps
conversation history so you can ask follow-up questions in the same session.

Usage:
  python examples/vision-agent.py path/to/image.jpg "A red apple on a table"

Requirements:
  A vision-capable model must be pulled first, e.g.:
    ollama pull llava
    ollama pull gemma3   # or another multimodal model
"""

import sys

from ollama import Agent

SYSTEM_PROMPT = (
  'You are an expert image reviewer. '
  'Your job is to evaluate whether a given text description accurately matches the content of the provided image. '
  'Respond with a clear verdict — Correct or Incorrect — followed by a brief explanation of what you observed.'
)


def main() -> None:
  if len(sys.argv) < 3:
    print('Usage: python vision-agent.py <image_path> <description>')
    sys.exit(1)

  image_path = sys.argv[1]
  description = sys.argv[2]

  # No tools needed — this agent is purely a VLM conversational wrapper.
  agent = Agent(
    model='llava',
    system=SYSTEM_PROMPT,
  )

  # First turn: evaluate the description against the image
  response = agent.chat(
    f'Does this description match the image?\nDescription: {description}',
    images=[image_path],
  )
  print('Evaluation:', response.message.content)

  # The agent retains history so you can ask follow-up questions.
  followup = agent.chat('What specific details in the image led to your verdict?')
  print('Follow-up:', followup.message.content)


if __name__ == '__main__':
  main()
