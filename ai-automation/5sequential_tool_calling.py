from ollama import chat, ChatResponse

def add(a: int, b: int) -> int:
  return a + b

def multiply(a: int, b: int) -> int:
  return a * b

def subtract(a: int, b:int) -> int:
   return a - b

available_functions = {
  'add': add,
  'multiply': multiply,
  'subtract': subtract,
}

messages = [{'role': 'user', 'content': 'What is (11434+12341)*412-12?'}]
while True:
    response: ChatResponse = chat(
        model='qwen2.5:7b',
        messages=messages,
        tools=[add, multiply, subtract],
    )
    messages.append(response.message)
    print("Content: ", response.message.content)
    if response.message.tool_calls:
        for tc in response.message.tool_calls:
            if tc.function.name in available_functions:
                print(f"Calling {tc.function.name} with arguments {tc.function.arguments}")
                result = available_functions[tc.function.name](**tc.function.arguments)
                print(f"Result: {result}")
                # add the tool result to the messages
                messages.append({'role': 'tool', 'tool_name': tc.function.name, 'content': str(result)})
    else:
        break