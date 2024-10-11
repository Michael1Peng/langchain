from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PythonLoader

# Load markdown files
loader = DirectoryLoader("../", glob="**/*.md")
docs = loader.load()
print(f"Number of documents loaded: {len(docs)}")

print("First 100 characters of the first document:")
print(docs[0].page_content[:100])

# Show progress bar
loader = DirectoryLoader("../", glob="**/*.md", show_progress=True)
docs = loader.load()

# Use multithreading
loader = DirectoryLoader("../", glob="**/*.md", use_multithreading=True)
docs = loader.load()

# Change loader class to TextLoader
loader = DirectoryLoader("../", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()

print("First 100 characters of the first document using TextLoader:")
print(docs[0].page_content[:100])

# Load Python files
loader = DirectoryLoader("../../../../../", glob="**/*.py", loader_cls=PythonLoader)
docs = loader.load()

# Auto-detect file encodings with TextLoader
path = "../../../../libs/langchain/tests/unit_tests/examples/"

# A. Default Behavior
loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
try:
    docs = loader.load()
except RuntimeError as e:
    print(f"Error occurred: {str(e)}")

# B. Silent fail
loader = DirectoryLoader(
    path, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True
)
docs = loader.load()

doc_sources = [doc.metadata["source"] for doc in docs]
print("Documents loaded with silent errors:")
print(doc_sources)

# C. Auto detect encodings
text_loader_kwargs = {"autodetect_encoding": True}
loader = DirectoryLoader(
    path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs
)
docs = loader.load()

doc_sources = [doc.metadata["source"] for doc in docs]
print("Documents loaded with auto-detected encoding:")
print(doc_sources)
