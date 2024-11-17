# %%
# ---
# sidebar_label: OpenAI
# ---

# %% [markdown]
# # ChatOpenAI
# 
# This notebook provides a quick overview for getting started with OpenAI [chat models](/docs/concepts/#chat-models). For detailed documentation of all ChatOpenAI features and configurations head to the [API reference](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html).
# 
# OpenAI has several chat models. You can find information about their latest models and their costs, context windows, and supported input types in the [OpenAI docs](https://platform.openai.com/docs/models).
# 
# :::info Azure OpenAI
# 
# Note that certain OpenAI models can also be accessed via the [Microsoft Azure platform](https://azure.microsoft.com/en-us/products/ai-services/openai-service). To use the Azure OpenAI service use the [AzureChatOpenAI integration](/docs/integrations/chat/azure_chat_openai/).
# 
# :::

# %% [markdown]
# ## Overview
# 
# ### Integration details
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/openai) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatOpenAI](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html) | [langchain-openai](https://python.langchain.com/api_reference/openai/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-openai?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-openai?style=flat-square&label=%20) |
# 
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | Image input | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | 
# 
# ## Setup
# 
# To access OpenAI models you'll need to create an OpenAI account, get an API key, and install the `langchain-openai` integration package.
# 
# ### Credentials
# 
# Head to https://platform.openai.com to sign up to OpenAI and generate an API key. Once you've done this set the OPENAI_API_KEY environment variable:

# %%
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# %% [markdown]
# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# %%
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"

# %% [markdown]
# ### Installation
# 
# The LangChain OpenAI integration lives in the `langchain-openai` package:

# %%
# %pip install -qU langchain-openai

# %% [markdown]
# ## Instantiation
# 
# Now we can instantiate our model object and generate chat completions:

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

# %% [markdown]
# ## Invocation

# %%
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg

# %%
print(ai_msg.content)

# %% [markdown]
# ## Chaining
# 
# We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:

# %%
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)

# %% [markdown]
# ## Tool calling
# 
# OpenAI has a [tool calling](https://platform.openai.com/docs/guides/function-calling) (we use "tool calling" and "function calling" interchangeably here) API that lets you describe tools and their arguments, and have the model return a JSON object with a tool to invoke and the inputs to that tool. tool-calling is extremely useful for building tool-using chains and agents, and for getting structured outputs from models more generally.
# 
# ### ChatOpenAI.bind_tools()
# 
# With `ChatOpenAI.bind_tools`, we can easily pass in Pydantic classes, dict schemas, LangChain tools, or even functions as tools to the model. Under the hood these are converted to an OpenAI tool schemas, which looks like:
# ```
# {
#     "name": "...",
#     "description": "...",
#     "parameters": {...}  # JSONSchema
# }
# ```
# and passed in every model invocation.

# %%
from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])

# %%
ai_msg = llm_with_tools.invoke(
    "what is the weather like in San Francisco",
)
ai_msg

# %% [markdown]
# ### ``strict=True``
# 
# :::info Requires ``langchain-openai>=0.1.21rc1``
# 
# :::
# 
# As of Aug 6, 2024, OpenAI supports a `strict` argument when calling tools that will enforce that the tool argument schema is respected by the model. See more here: https://platform.openai.com/docs/guides/function-calling
# 
# **Note**: If ``strict=True`` the tool definition will also be validated, and a subset of JSON schema are accepted. Crucially, schema cannot have optional args (those with default values). Read the full docs on what types of schema are supported here: https://platform.openai.com/docs/guides/structured-outputs/supported-schemas. 

# %%
llm_with_tools = llm.bind_tools([GetWeather], strict=True)
ai_msg = llm_with_tools.invoke(
    "what is the weather like in San Francisco",
)
ai_msg

# %% [markdown]
# ### AIMessage.tool_calls
# Notice that the AIMessage has a `tool_calls` attribute. This contains in a standardized ToolCall format that is model-provider agnostic.

# %%
ai_msg.tool_calls

# %% [markdown]
# For more on binding tools and tool call outputs, head to the [tool calling](/docs/how_to/function_calling) docs.

# %% [markdown]
# ## Fine-tuning
# 
# You can call fine-tuned OpenAI models by passing in your corresponding `modelName` parameter.
# 
# This generally takes the form of `ft:{OPENAI_MODEL_NAME}:{ORG_NAME}::{MODEL_ID}`. For example:

# %%
fine_tuned_model = ChatOpenAI(
    temperature=0, model_name="ft:gpt-3.5-turbo-0613:langchain::7qTVM5AR"
)

fine_tuned_model.invoke(messages)

# %% [markdown]
# ## API reference
# 
# For detailed documentation of all ChatOpenAI features and configurations head to the API reference: https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html


