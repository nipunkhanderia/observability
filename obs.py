from langfuse import Langfuse, observe
from dotenv import load_dotenv

load_dotenv()



@observe
def llm_response():
    return ("Nipun rocks")

llm_response()


