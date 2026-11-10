from ollama import chat
from pathlib import Path
import requests

def get_temperature(city: str) -> str:
  temperatures = {
    "New York": "22°C",
    "London": "15°C",
    "Tokyo": "18°C",
  } #you can replace temperatures with pulling from a weather api
  return temperatures.get(city, "Unknown")

messages = [{"role": "user", "content": "What is the temperature in New York?"}]

# pass functions directly as tools in the tools list or as a JSON schema
response = chat(model="llama3.2:1b", messages=messages, tools=[get_temperature])

messages.append(response.message)
if response.message.tool_calls:
  # only recommended for models which only return a single tool call
  call = response.message.tool_calls[0]
  result = get_temperature(**call.function.arguments)
  # add the tool result to the messages
  messages.append({"role": "tool", "tool_name": call.function.name, "content": str(result)})

  final_response = chat(model="llama3.2:1b", messages=messages, tools=[get_temperature])
  print(final_response.message.content)