from langfuse import Langfuse, observe
from dotenv import load_dotenv

load_dotenv()

langfuse = Langfuse()

@observe
def llm_response(input):
    langfuse.update_current_generation(model = "llama3.2", metadata = {"temperature":0.7,"max_tokens":100})
    return ("Nipun rocks")

llm_response("Who is Nipun and what does he do?")


