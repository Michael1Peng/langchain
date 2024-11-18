import getpass
import os

os.environ["ANTHROPIC_API_URL"] = os.getenv(
    "ANTHROPIC_API_URL", "https://api.deepbricks.ai"
)
os.environ["ANTHROPIC_BASE_URL"] = os.getenv(
    "ANTHROPIC_BASE_URL", "https://api.deepbricks.ai"
)

os.environ["ANTHROPIC_API_KEY"] = os.getenv(
    "ANTHROPIC_API_KEY", "sk-TwgrmzcdN4SOHjeVoEaXfmhTyNCsm9BLczas6V9W5y7PJGqJ"
)

print("Anthropic API URL:", os.environ["ANTHROPIC_API_URL"])
print("Anthropic Base URL:", os.environ["ANTHROPIC_BASE_URL"])
print("Anthropic API Key:", os.environ["ANTHROPIC_API_KEY"])


from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="Claude-3.5-Sonnet",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg

print(ai_msg.content)


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


from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])
ai_msg = llm_with_tools.invoke("Which city is hotter today: LA or NY?")
ai_msg.content

ai_msg.tool_calls
