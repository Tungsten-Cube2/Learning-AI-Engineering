from ollama import chat

#if tool calling doesn't work, just get a better ai model
def pull_message(message_num:int) -> str:
    message = {1: "hello", 2: "hello world", 3: "something", 5: "why is the sky blue. in one paragraph?",}
    return message.get(message_num, "unknown")

messages = [
    {"role": "assistant", "content":"You are a puller of messages from a tool. Retrieve the message from the tool by pulling from the tool, not from your chat history. After obtaining the message, DO NOT respond to it."},
    {"role": "user", "content": "What is message 1? ONLY SAY THE MESSAGE AND DO NOT SAY ANYTHING ELSE, AND DO NOT ANSWER."}]
response = chat(model="qwen2.5:7b", messages=messages, tools=[pull_message])

print(response.message)
messages.append(response.message)

if response.message.tool_calls:
    call = response.message.tool_calls[0]
    result = pull_message(**call.function.arguments)
    messages.append({"role": "tool", "tool_name": call.function.name, "content": str(result)})

    final_response = chat(model="qwen2.5:7b", messages=messages, tools=[pull_message])
    print(final_response.message.content)