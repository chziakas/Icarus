from llama_index.schema import Document
from llama_index.agent import ContextRetrieverOpenAIAgent
from llama_index import VectorStoreIndex

from llama_index import SummaryIndex, SimpleDirectoryReader
from dotenv import load_dotenv
load_dotenv()

documents = SimpleDirectoryReader('data').load_data()
index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine(response_mode="tree_summarize")

response = query_engine.query("What is my name?")

print(response)