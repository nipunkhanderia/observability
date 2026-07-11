"""
Semantic Cache Example with Ollama (llama3.2)
Install: pip install scikit-learn numpy langchain-ollama
"""
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama
from langfuse import get_client, propagate_attributes
from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()

CACHE_FILE = "semantic_cache.json"
SIMILARITY_THRESHOLD = 0.7

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
else:
    cache = []

llm = ChatOllama(model="llama3.2")

def find_similar(prompt):
    if not cache:
        return None
    stored_prompts = [item["prompt"] for item in cache]
    all_prompts = stored_prompts + [prompt]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_prompts)

    similarities = cosine_similarity(vectors[-1:], vectors[:-1])[0]
    best_idx = np.argmax(similarities)

    if similarities[best_idx] >= SIMILARITY_THRESHOLD:
        return cache[best_idx]["response"]
    return None

def ask(prompt):
    cached = find_similar(prompt)

    if cached:
        with propagate_attributes(trace_name="FROM CACHE"):
            with langfuse.start_as_current_observation(name="cache-lookup", as_type="span") as span:
                span.update(input=prompt, output=cached, metadata={"source": "cache"})
        print("(from cache)")
        return cached
    else:
        with propagate_attributes(trace_name="FROM LLM"):
            with langfuse.start_as_current_observation(name="llm-call", as_type="generation") as span:
                response = llm.invoke(prompt).content  # LLM call INSIDE the span
                span.update(input=prompt, output=response, model="llama3.2", metadata={"source": "llm"})
        cache.append({"prompt": prompt, "response": response})
        print("(from LLM)")
        return response

# First call
print(ask("What is AI? Explain briefly"))

# Second call - similar prompt, should hit cache
# print(ask("what are countries? brief answer"))

# Save cache
with open(CACHE_FILE, "w") as f:
    json.dump(cache, f, indent=2)

langfuse.flush()
