# %%
# ---
# sidebar_label: Anthropic
# ---

# %% [markdown]
# # ChatAnthropic
#
# This notebook provides a quick overview for getting started with Anthropic [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatAnthropic features and configurations head to the [API reference](https://python.langchain.com/api_reference/anthropic/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html).
#
# Anthropic has several chat models. You can find information about their latest models and their costs, context windows, and supported input types in the [Anthropic docs](https://docs.anthropic.com/en/docs/models-overview).
#
#
# :::info AWS Bedrock and Google VertexAI
#
# Note that certain Anthropic models can also be accessed via AWS Bedrock and Google VertexAI. See the [ChatBedrock](/docs/integrations/chat/bedrock/) and [ChatVertexAI](/docs/integrations/chat/google_vertex_ai_palm/) integrations to use Anthropic models via these services.
#
# :::
#
# ## Overview
# ### Integration details
#
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/anthropic) | Package downloads | Package latest |
# | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
# | [ChatAnthropic](https://python.langchain.com/api_reference/anthropic/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html) | [langchain-anthropic](https://python.langchain.com/api_reference/anthropic/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-anthropic?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-anthropic?style=flat-square&label=%20) |
#
# ### Model features
# | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
# | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
# | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
#
# ## Setup
#
# To access Anthropic models you'll need to create an Anthropic account, get an API key, and install the `langchain-anthropic` integration package.
#
# ### Credentials
#
# Head to https://console.anthropic.com/ to sign up for Anthropic and generate an API key. Once you've done this set the ANTHROPIC_API_KEY environment variable:

# %%
import getpass
import os

# Set the Anthropic API URL and base URL from environment variables
os.environ["ANTHROPIC_API_URL"] = os.getenv(
    "ANTHROPIC_API_URL", "https://api.deepbricks.ai"
)
os.environ["ANTHROPIC_BASE_URL"] = os.getenv(
    "ANTHROPIC_BASE_URL", "https://api.deepbricks.ai"
)

# Set the Anthropic API key from the environment variable
os.environ["ANTHROPIC_API_KEY"] = os.getenv(
    "ANTHROPIC_API_KEY", "sk-TwgrmzcdN4SOHjeVoEaXfmhTyNCsm9BLczas6V9W5y7PJGqJ"
)

# Print a message to confirm the environment variables are set
print("Anthropic API URL:", os.environ["ANTHROPIC_API_URL"])
print("Anthropic Base URL:", os.environ["ANTHROPIC_BASE_URL"])
print("Anthropic API Key:", os.environ["ANTHROPIC_API_KEY"])

# os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")

# %% [markdown]
# If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# %%
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"

# %% [markdown]
# ### Installation
#
# The LangChain Anthropic integration lives in the `langchain-anthropic` package:

# %%
# %pip install -qU langchain-anthropic

# %% [markdown]
# ## Instantiation
#
# Now we can instantiate our model object and generate chat completions:

# %%
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="Claude-3.5-Sonnet",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    # other params...
)

# %% [markdown]
# ## Invocation
#

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
# ## Content blocks
#
# One key difference to note between Anthropic models and most others is that the contents of a single Anthropic AI message can either be a single string or a **list of content blocks**. For example when an Anthropic model invokes a tool, the tool invocation is part of the message content (as well as being exposed in the standardized `AIMessage.tool_calls`):

# %%
from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])
ai_msg = llm_with_tools.invoke("Which city is hotter today: LA or NY?")
ai_msg.content

# %%
ai_msg.tool_calls

# %% [markdown]
# ## API reference
#
# For detailed documentation of all ChatAnthropic features and configurations head to the API reference: https://python.langchain.com/api_reference/anthropic/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html
