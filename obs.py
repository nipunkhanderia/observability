from langfuse import Langfuse, observe
from dotenv import load_dotenv

from langchain_ollama import ChatOllama

load_dotenv()

import json
import os

if os.path.exists("cache.json"):
    with open("cache.json", "r") as f:
        cache = json.load(f)
else:
    cache = {}

langfuse = Langfuse()



@observe
def invokation_llm(input):
    llm = ChatOllama(model = "llama3.2")
    # langfuse.update_current_generation(model = "llama3.2", metadata = {"temperature":0.7,"max_tokens":100})
    langfuse.update_current_generation(model = "Claude Fable", metadata = {"temparature": 0.4, "env":"qa"})
    if input in cache:
        response = cache[input]
    else:
        response = llm.invoke(input).content
        print(response)
        cache[input] = response
        return response

# input = "Why is ocean blue? tell me it in small sentence."
prompt = ""
while prompt != "exit":
    prompt = input("Enter prompt")
    invokation_llm(prompt)
    print(f"=============The stored response is ======{cache[prompt]}")
    print(cache)

with open("cache.json", "w") as f:
    json.dump(cache, f, indent =2)

# @observe
# def llm_response(input):
#     langfuse.update_current_generation(model = "llama3.2", metadata = {"temperature":0.7,"max_tokens":100})
#     return ("Nipun rocks")

# llm_response("Who is Nipun and what does he do?")


