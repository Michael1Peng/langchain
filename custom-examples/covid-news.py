from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

DB_USER = "wsl_user"
DB_PASSWORD = "a124567"
DB_HOST = "localhost"
DB_NAME = "media_crawler"


engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
)

db = SQLDatabase(engine=engine)
print(db.dialect)
print(db.get_usable_table_names())
print(db.run("SELECT * FROM xhs_note LIMIT 10;"))

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langchain_community.agent_toolkits import create_sql_agent

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# response = agent_executor.invoke({"input": "how many notes with content '旅游' are there in xhs_note"})
# print(response)

response = agent_executor.invoke({"input": "summarize the news about '新冠' in xhs_note"})
print(response)
