import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.globals import set_verbose, set_debug

set_verbose(True)
set_debug(True)

# Set up the file path for the reference posts
reference_folder = "/home/michael/ubuntu-repos/langchain/custom-examples/xhs-notes"

# Load and process the reference posts
loader = DirectoryLoader(reference_folder, glob="**/*.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create a vector store from the documents
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

# Set up the language model
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini", verbose=True)

# Create a prompt template for generating new posts
template = """
You are a Xiaohongshu (小红书) content creator. Your task is to create a new post based on the style and content of existing posts. 
Here are some example posts for reference:

{context}

Using these examples as inspiration, create a new Xiaohongshu post on the topic below. The post should be engaging, informative, and in a style typical of Xiaohongshu content.

New Xiaohongshu Topic:
{topic}
"""

prompt = PromptTemplate(
    input_variables=["context", "topic"],
    template=template,
)

# Define the document formatting function
def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        title = doc.metadata.get('title', 'Untitled')
        content = doc.page_content.strip()
        formatted_doc = f"<post>\n<title>{title}</title>\n<content>\n{content}\n</content>\n</post>"
        formatted_docs.append(formatted_doc)
    return "\n\n".join(formatted_docs)

# Set up the retriever
retriever = db.as_retriever(search_kwargs={"k": 10}, verbose=True)

# Create the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "topic": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Generate a new post
result = rag_chain.invoke("北京旅游", verbose=True)

print("Generated Xiaohongshu Post:")
print(result)
