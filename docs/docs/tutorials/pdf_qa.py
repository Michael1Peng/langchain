import subprocess
import sys

def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-qU", "pypdf", "langchain_community", "langchain_openai"])

# install_packages()

from langchain_community.document_loaders import PyPDFLoader

file_path = "/home/michael/ubuntu-repos/langchain/docs/docs/example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print(len(docs))

print(docs[0].page_content[0:100])
print(docs[0].metadata)

import getpass
import os
from langchain_anthropic import ChatAnthropic

if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Anthropic API Key:")

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

import subprocess
import sys

def install_langchain_openai():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain_openai"])

# install_langchain_openai()

import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": "What was Nike's revenue in 2023?"})

print(results)

print(results["context"][0].page_content)
print(results["context"][0].metadata)
