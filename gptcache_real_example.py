"""
GPTCache Example with Ollama (llama3.2)
Install in a SEPARATE venv to avoid conflicts:
    python -m venv gptcache-env
    gptcache-env/Scripts/activate
    pip install gptcache langchain-ollama onnxruntime faiss-cpu transformers==4.44.0

Run: python gptcache_real_example.py
"""
from gptcache import cache
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from langchain_ollama import ChatOllama
from langfuse import get_client, propagate_attributes
from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()

# Setup GPTCache
onnx = Onnx()
cache_base = CacheBase("sqlite")
vector_base = VectorBase("faiss", dimension=onnx.dimension)
data_manager = get_data_manager(cache_base, vector_base)

cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)

llm = ChatOllama(model="llama3.2")

def ask(prompt):
    # Search for similar cached prompt
    embedding = onnx.to_embeddings(prompt)
    search_result = data_manager.search(embedding)

    if search_result and search_result[0][0] <= 0.2:
        cached_id = search_result[0][1]
        cached_data = data_manager.s.get_data_by_id(cached_id)
        response = cached_data.answers[0].answer
        with propagate_attributes(trace_name="FROM CACHE (GPTCache)"):
            with langfuse.start_as_current_observation(name="cache-lookup", as_type="span") as span:
                span.update(input=prompt, output=response, metadata={"source": "gptcache", "distance": float(search_result[0][0])})
        print("(from GPTCache - semantic match)")
        return response

    # Cache miss - call Ollama
    with propagate_attributes(trace_name="FROM LLM"):
        with langfuse.start_as_current_observation(name="llm-call", as_type="generation") as span:
            response = llm.invoke(prompt).content
            span.update(input=prompt, output=response, model="llama3.2", metadata={"source": "llm"})
    data_manager.save(prompt, response, embedding)
    print("(from LLM)")
    return response

# First call - hits LLM
print(ask("continents?"))
print("---")

# Second call - similar prompt, should hit cache
# print(ask("Why does the sky appear blue?"))

langfuse.flush()
