from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
def main():
    # Initialize the vector store with Chroma
    vector_store = Chroma(
        persist_directory="./chroma",
        embedding_function=OpenAIEmbeddings()  # Replace with your embedding function
    )

    # Initialize the QA chain with LangChain's RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=vector_store.as_retriever()
    )

    # Sample input
    input_text = "What is the capital of France?"

    # Generate answer using the QA chain
    answer = qa_chain.run(input_text)

    # Print the answer
    print(f"Question: {input_text}")
    print(f"Answer: {answer}")
import os
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def initialize_vector_store():
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Load sample documents
    loader = TextLoader("/home/michael/ubuntu-repos/langchain/custom-examples/xhs-notes/一个人旅游连拍视频都可后置自拍镜①全身照机位放在胸前的位置镜头微微抬起来一点点把人放在下面的两层脚尖.txt")  # Replace with your document path
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    # Initialize Chroma vector store and add documents
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./chroma"  # Replace with your desired persist directory
    )

    return vector_store

def main():
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Initialize the vector store
    vector_store = initialize_vector_store()

    # Initialize the QA chain with LangChain's RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(model="gpt-4o"),
        retriever=vector_store.as_retriever()
    )

    # Sample input
    input_text = "What is the capital of France?"

    # Generate answer using the QA chain
    answer = qa_chain.run(input_text)

    # Print the answer
    print(f"Question: {input_text}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
