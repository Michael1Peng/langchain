# %% [markdown]
# # How to split code
# 
# [RecursiveCharacterTextSplitter](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html) includes pre-built lists of separators that are useful for splitting text in a specific programming language.
# 
# Supported languages are stored in the `langchain_text_splitters.Language` enum. They include:
# 
# ```
# "cpp",
# "go",
# "java",
# "kotlin",
# "js",
# "ts",
# "php",
# "proto",
# "python",
# "rst",
# "ruby",
# "rust",
# "scala",
# "swift",
# "markdown",
# "latex",
# "html",
# "sol",
# "csharp",
# "cobol",
# "c",
# "lua",
# "perl",
# "haskell"
# ```
# 
# To view the list of separators for a given language, pass a value from this enum into
# ```python
# RecursiveCharacterTextSplitter.get_separators_for_language`
# ```
# 
# To instantiate a splitter that is tailored for a specific language, pass a value from the enum into
# ```python
# RecursiveCharacterTextSplitter.from_language
# ```
# 
# Below we demonstrate examples for the various languages.

# %%
# %pip install -qU langchain-text-splitters

# %%
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

# %% [markdown]
# To view the full list of supported languages:

# %%
[e.value for e in Language]

# %% [markdown]
# You can also see the separators used for a given language:

# %%
RecursiveCharacterTextSplitter.get_separators_for_language(Language.TS)

# %% [markdown]
# ## Python
# 
# Here's an example using the PythonTextSplitter:
# 
# 

# %%
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
python_docs

# %% [markdown]
# ## JS
# Here's an example using the JS text splitter:

# %%
JS_CODE = """

"""

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=600, chunk_overlap=0
)
js_docs = js_splitter.create_documents([JS_CODE])
js_docs

# %% [markdown]
# ## TS
# Here's an example using the TS text splitter:

# %%
TS_CODE = """
function helloWorld(): void {
  console.log("Hello, World!");
}

// Call the function
helloWorld();
"""

ts_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.TS, chunk_size=60, chunk_overlap=0
)
ts_docs = ts_splitter.create_documents([TS_CODE])
ts_docs

# %% [markdown]
# ## Markdown
# 
# Here's an example using the Markdown text splitter:
# 

# %%
markdown_text = """
# 🦜️🔗 LangChain

⚡ Building applications with LLMs through composability ⚡

## Quick Install

# Hopefully this code block isn't split
pip install langchain

As an open-source project in a rapidly developing field, we are extremely open to contributions.
"""

# %%
md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
)
md_docs = md_splitter.create_documents([markdown_text])
md_docs

# %% [markdown]
# ## Latex
# 
# Here's an example on Latex text:
# 

# %%
latex_text = """
\documentclass{article}

\begin{document}

\maketitle

\section{Introduction}
Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in a variety of natural language processing tasks, including language translation, text generation, and sentiment analysis.

\subsection{History of LLMs}
The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.

\subsection{Applications of LLMs}
LLMs have many applications in industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.

\end{document}
"""

# %%
latex_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
)
latex_docs = latex_splitter.create_documents([latex_text])
latex_docs

# %% [markdown]
# ## HTML
# 
# Here's an example using an HTML text splitter:
# 

# %%
html_text = """
<!DOCTYPE html>
<html>
    <head>
        <title>🦜️🔗 LangChain</title>
        <style>
            body {
                font-family: Arial, sans-serif;
            }
            h1 {
                color: darkblue;
            }
        </style>
    </head>
    <body>
        <div>
            <h1>🦜️🔗 LangChain</h1>
            <p>⚡ Building applications with LLMs through composability ⚡</p>
        </div>
        <div>
            As an open-source project in a rapidly developing field, we are extremely open to contributions.
        </div>
    </body>
</html>
"""

# %%
html_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HTML, chunk_size=60, chunk_overlap=0
)
html_docs = html_splitter.create_documents([html_text])
html_docs

# %% [markdown]
# ## Solidity
# Here's an example using the Solidity text splitter:

# %%
SOL_CODE = """
pragma solidity ^0.8.20;
contract HelloWorld {
   function add(uint a, uint b) pure public returns(uint) {
       return a + b;
   }
}
"""

sol_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.SOL, chunk_size=128, chunk_overlap=0
)
sol_docs = sol_splitter.create_documents([SOL_CODE])
sol_docs

# %% [markdown]
# ## C#
# Here's an example using the C# text splitter:
# 

# %%
C_CODE = """
using System;
class Program
{
    static void Main()
    {
        int age = 30; // Change the age value as needed

        // Categorize the age without any console output
        if (age < 18)
        {
            // Age is under 18
        }
        else if (age >= 18 && age < 65)
        {
            // Age is an adult
        }
        else
        {
            // Age is a senior citizen
        }
    }
}
"""
c_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.CSHARP, chunk_size=128, chunk_overlap=0
)
c_docs = c_splitter.create_documents([C_CODE])
c_docs

# %% [markdown]
# ## Haskell
# Here's an example using the Haskell text splitter:

# %%
HASKELL_CODE = """
main :: IO ()
main = do
    putStrLn "Hello, World!"
-- Some sample functions
add :: Int -> Int -> Int
add x y = x + y
"""
haskell_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HASKELL, chunk_size=50, chunk_overlap=0
)
haskell_docs = haskell_splitter.create_documents([HASKELL_CODE])
haskell_docs

# %% [markdown]
# ## PHP
# Here's an example using the PHP text splitter:

# %%
PHP_CODE = """<?php
namespace foo;
class Hello {
    public function __construct() { }
}
function hello() {
    echo "Hello World!";
}
interface Human {
    public function breath();
}
trait Foo { }
enum Color
{
    case Red;
    case Blue;
}"""
php_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PHP, chunk_size=50, chunk_overlap=0
)
php_docs = php_splitter.create_documents([PHP_CODE])
php_docs

# %% [markdown]
# ## PowerShell
# Here's an example using the PowerShell text splitter:

# %%
POWERSHELL_CODE = """
$directoryPath = Get-Location

$items = Get-ChildItem -Path $directoryPath

$files = $items | Where-Object { -not $_.PSIsContainer }

$sortedFiles = $files | Sort-Object LastWriteTime

foreach ($file in $sortedFiles) {
    Write-Output ("Name: " + $file.Name + " | Last Write Time: " + $file.LastWriteTime)
}
"""
powershell_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.POWERSHELL, chunk_size=100, chunk_overlap=0
)
powershell_docs = powershell_splitter.create_documents([POWERSHELL_CODE])
powershell_docs


