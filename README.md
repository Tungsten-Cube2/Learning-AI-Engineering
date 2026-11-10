# Learning-AI-Engineering
This is going to show all of my progress on learning AI engineering. On the way, I might just write how to learn all of this on this repository...

## Prereqs
If you want to start AI engineering, firstly, you should learn basic Python...

Here's some courses and videos that should help you:
Harvard CS50P: [https://youtube.com/playlist?list=PLhQjrBD2T3817j24-GogXmWqO5Q5vYy0V&si=HmdJFVKQBAmG4hlu](url)

Bro Code's Python Playlist: [https://www.youtube.com/watch?v=Sg4GMVMdOPo&list=PLZPZq0r_RZOOkUQbat8LyQii36cJf2SWT](url)

## Starting with AI APIs
After learning all about Python (hopefully you learned about APIs), then it's time to start with AI basics.

### Installing Ollama
To start using AI APIs locally and for free, start off with using Ollama. It lets you install AI models directly onto your computer and lets you run them locally. It also has support for Hugging Face models, though it isn't necessary.

To install Ollama, go here: [https://ollama.com/](url)

Later, you can use OpenAI or Anthropic API, but for now, use this.

### Your First Program
Now with an AI model installed onto your computer, you can finally start making your first program using Ollama.

Firstly, install the ollama package by doing this:
```console
pip install ollama
```

Next after installing the ollama package, install an AI model from here: [https://ollama.com/search](url) 

After that, you can finally write code that calls an AI model (note that you don't need an API key for now). It should look something like this:

```python
import ollama

stream = ollama.chat(
    model='llama3.2:1b',
    messages=[{'role': 'user', 'content': "Explain why programmers use 'hello world' in one paragraph."}],
)

print(stream.message['content'])
```
Now, you have a basic Ollama chatbot. This is where you have officialy started your AI engineering adventure. To continue, follow this: [https://docs.ollama.com/capabilities/streaming](url)

#### A beginner's chatbot...
Finally, after learning about basic Ollama basics, it's finally time to do one last thing before continuing on.

**Make a chatbot that has memory**

Now this is actually pretty simple. All you do is track what you and the bot said, and then you track it via a nested data structure, which is used in the chat history. Bonus points if you can save the chat history to a file and remove it via a command.

<details>
<summary>Here is my answer to this problem: </summary>

```python
import ollama
import time
import keyboard
import os
import json

ai_model = 'qwen2.5:7b'
def main():
    clearterminal()
    with open("ai_history.json", "r", encoding="utf-8") as file:
        totalmessagelist = file.read()
    if totalmessagelist == "":
        totalmessagelist = []
    else:
        totalmessagelist = [item for item in json.loads(totalmessagelist)]
    while True:
        ai_inp = input("Ask the AI anything (enter '/help' for commands): ")
        try:
            response_quitted = False
            response_written = []
            ai_message = {'role': 'user', 'content': ai_inp}
            a = ""
            if ai_inp.find("/") == 0:
                if ai_inp == "/quit":
                    break
                elif ai_inp == "/help":
                    print("""Commands:
                    /quit: quits the program
                    /clearhistory: clears response history
                    /read: reads chat history""")
                elif ai_inp == "/clearhistory":
                    clearterminal()
                    with open("ai_response.txt", "w", encoding="utf-8") as file:
                        file.write("")
                    with open("ai_history.json", "w", encoding="utf-8") as file:
                        file.write("")
                        totalmessagelist = []
                    print("history cleared")
                elif ai_inp == "/read":
                    with open("ai_response.txt", "r", encoding="utf-8") as file:
                        read = file.read()
                    print(read)
                else:
                    print("Command doesn't exist.")
            else:
                with open("ai_response.txt", "a", encoding="utf-8") as file:
                    file.write("User: " + ai_inp + "\n")
                totalmessagelist.append(ai_message)
                print("Generating response...")
                current_time = time.time() #start of timer
                stream = ollama.chat(
                    model=ai_model,
                    messages=totalmessagelist,
                    options={'num_ctx': 8192,},
                    stream=True,
                )
                #what stream=True does is lets you see the response being generated in real time
                #instead of waiting for the entire response to be generated before you see it
                for chunk in stream: #This loop goes through each chunk of the response as it's generated
                    print(chunk['message']['content'], end='', flush=True)
                    response_written.append(chunk['message']['content'])
                    if keyboard.is_pressed('esc'):
                        print("\nResponse has been quitted.")
                        response_quitted = True
                        response_written.clear()
                        break
                print("\n")
    
                if response_quitted == False:
                    x = "".join(response_written)
                    y = ''.join(response_written)
                    a = {'role': 'assistant', 'content': y}
                    totalmessagelist.append(a)
                    with open("ai_response.txt", "a", encoding="utf-8") as file:
                        file.write("Assistant: " + x + "\n")
                    x = ""
                    with open("ai_history.json", "w", encoding="utf-8") as file:
                        file.write(json.dumps(totalmessagelist, indent=4))
                #time finished, now we calculate the time taken for the response to be generated
                current_time = time.time() - current_time
                print(f"Time taken: ~{round(current_time, 2)} seconds")
        except KeyboardInterrupt:
            clearterminal()
            print("\n ctrl+c will not work. Please use /quit to quit.")

def clearterminal():
    os.system('cls' if os.name == 'nt' else 'clear')

main()
```
</details>
So, after creating a basic chatbot that can remember previous chat history, it will be time to move on...

### AI Automation
To learn AI automation, you will be learning:
- Structured Output
- Tool Calling
- API Integration
- Asyncronization
- AI Workflows and Orchestration
- Basic Embeddings/RAG (for optimization)

#### Structured Output
Structured output lets you receive JSON for any specific data. With the JSON, you can do whatever you want. This is done by using the Pydantic library's BaseModel.

To get started, import pydantic and BaseModel, and use it to build a class:
```python
from pydantic import BaseModel

class Country(BaseModel):
  name: str #these str fields get filled out by the AI
  short_description: str
  absolute_location: str
  economy_type: str
  states_provinces: list[str] #lets you have multiple strings
  territories: list[str]
```
If you need multiple instances of a class, you can also do this:
```python
class CountryList(BaseModel):
    countries: list[Country]
```
Now countries can be listed and used by ollama:
```python
countries = input("Enter countries here: ")
response = chat(
  model='llama3.2:1b',
  messages=[{'role': 'user', 'content': f"Tell me about {countries}. For each country, give me a short description, an absolute location, the economy type, all states or provinces, and owned territories."}],
  format=CountryList.model_json_schema(),
)

countrylist = CountryList.model_validate_json(response.message.content)
print(countrylist)
```
From there, you can do whatever you want with the data.
```python
#you can iterate though country names
for country in countrylist.countries:
  print(country.name)
```
Very good for data processing if needed.
#### Tool Calling
Tool calling lets you call functions from a script. This is useful if you need to:
- Actively call data from an API
- Perform actions
- Do things beyond the ai's training
This is done by, firstly, having a function that can do something and return a value.
```python
#this is our first function
def get_temperature(city: str) -> str:
  temperatures = {
    "New York": "22°C",
    "London": "15°C",
    "Tokyo": "18°C",
  } #you can replace temperatures with pulling from a weather api
  return temperatures.get(city, "Unknown")
```
After you have the function, the ai model can be called upon to use the tool when needed.
```python
messages = [{"role": "user", "content": "What is the temperature in New York?"}]
response = chat(model="llama3.2:1b", messages=messages, tools=[get_temperature])
#you then can use the call the AI model made to get the temperature for the temperature
call = response.message.tool_calls[0]
result = get_temperature(**call.function.arguments)
```
Finally, you can add the result to the messages.
```python
messages.append({"role": "tool", "tool_name": call.function.name, "content": str(result)})

final_response = chat(model="llama3.2:1b", messages=messages, tools=[get_temperature])
print(final_response.message.content)
```
# W.I.P.