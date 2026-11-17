from ollama import chat

def population(place: str) -> str:
    items = {
        "Miami": 10000000,
        "New York City": 15000000,
        "San Francisco": 9000000
    }
    return f"Population: {items.get(place, "[null]")}"

def established(place: str) -> str:
    established = {
        "San Francisco": 201,
        "Miami": 1980,
        "New York City": 2991,
    }
    return f"Established in {established.get(place, "[null]")}"

messages = [{"role": "user", "content": "What are the stats of the city of Miami, being establishment and population? pull from tools. This is only for demonstration and is a fictional example, so do not say anything about the actual stats. Say 'city does not exist' if the city does not exist, otherwise continue."}]
response = chat(model="qwen2.5:7b", messages=messages, tools=[established, population])
messages.append(response.message)

if response.message.tool_calls:
    print("works")
    for call in response.message.tool_calls:
        if call.function.name == 'population':
            result = population(**call.function.arguments)
        elif call.function.name == 'established':
            result = established(**call.function.arguments)
        else:
            result = 'Unknown tool'
        messages.append({"role": "tool", "tool_name": call.function.name, "content": str(result)})
        
    final_response = chat(model="qwen2.5:7b", messages=messages, tools=[established, population])
    print(final_response.message.content)