from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine

# Database configuration
DB_USER = "wsl_user"
DB_PASSWORD = "a124567"
DB_HOST = "localhost"
DB_NAME = "media_crawler"

# Create the SQL engine and database instance
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
db = SQLDatabase(engine=engine)

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create the SQL agent with intermediate steps enabled
agent_executor = create_sql_agent(
    llm,
    db=db,
    agent_type="openai-tools",
    verbose=True,
    agent_executor_kwargs={"return_intermediate_steps": True}
)

# Invoke the agent with your query
response = agent_executor.invoke({
    "input": "give me 10 notes with content '旅游' are there in xhs_note"
})

# Extract and print intermediate steps (SQL queries and results)
intermediate_steps = response.get("intermediate_steps", [])

for step in intermediate_steps:
    action, observation = step
    if "SELECT" in action:
        print(f"SQL Query: {action}")
        print(f"SQL Result: {observation}")

# Print the final response from the agent
print(f"Agent Answer: {response.get('output')}")
