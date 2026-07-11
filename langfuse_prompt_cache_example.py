"""
Langfuse Prompt Cache Example - Cache prompt templates from Langfuse server
Install: pip install langfuse
Setup: Create a prompt in Langfuse dashboard first (Settings > Prompts > New)
"""
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

langfuse = Langfuse()

# Fetch a prompt template from Langfuse (cached locally after first fetch)
# Create this prompt in your Langfuse dashboard first with a template like:
# "Answer the following question in one sentence: {{question}}"
prompt = langfuse.get_prompt("my-prompt", cache_ttl_seconds=300)

# Compile with variables
compiled = prompt.compile(question="Why is the sky blue?")
print("Compiled prompt:", compiled)

# Use it with your LLM
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")
response = llm.invoke(compiled).content
print("Response:", response)

langfuse.flush()
