from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

langfuse = Langfuse()

trace = langfuse.trace(name = "First trace", user_id = 1)

gen = trace.generation(
    input = [{"role" : "user", "content": "what is observability?"}],
    name = "LLM call",
    output = "This is how llm works",
    model = "llama3.2",
    metadat = {"temperature": 0.7}
)

trace.score()


