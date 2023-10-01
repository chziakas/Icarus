
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate

from llama_index import SummaryIndex, SimpleDirectoryReader
from dotenv import load_dotenv
load_dotenv()


llm = OpenAI(model="gpt-3.5-turbo-16k")

documents = SimpleDirectoryReader('data').load_data()
index = SummaryIndex.from_documents(documents)

retriever = index.as_retriever()

"""#### Given an example question, get a retrieved set of nodes.

We use the retriever to get a set of relevant nodes given a user query. These nodes will then be passed to the response synthesis modules below.
"""




qa_prompt = PromptTemplate(
    """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: \
"""
)

def generate_response_text(retrieved_nodes, query_str, qa_prompt, llm):
    context_str = "\n\n".join([r.get_content() for r in retrieved_nodes])
    fmt_qa_prompt = qa_prompt.format(context_str=context_str, query_str=query_str)
    response = llm.complete(fmt_qa_prompt)
    return str(response), fmt_qa_prompt


qa_prompt = PromptTemplate(
    """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, propose an action based on your previous memories.
Query: {query_str}
Answer: \
"""
)


query_str = "What should I do for fire prevention?"



retrieved_nodes = retriever.retrieve(query_str)
response, fmt_qa_prompt = generate_response_text(retrieved_nodes, query_str, qa_prompt, llm)
print(response)