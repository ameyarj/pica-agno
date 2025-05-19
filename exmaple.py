from langchain_openai import ChatOpenAI
from langchain.agents import AgentType
from pica_langchain import PicaClient, create_pica_agent
from pica_langchain.models import PicaClientOptions

from dotenv import load_dotenv
import os

load_dotenv()
PICA_SECRET = os.getenv("PICA_SECRET")
development_server_url = "https://development-api.picaos.com"

pica_client = PicaClient(secret=PICA_SECRET, options=PicaClientOptions(
                connectors=["*"], server_url=development_server_url
            ))

llm = ChatOpenAI(temperature=0, model="gpt-4o")

agent = create_pica_agent(
    client=pica_client,
    llm=llm,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

task = (
    "Star emails from moe@picaos.com and list 5 of the starred emails. "
    "Then, send a reply to the starred emails with the subject 'Hello' and the body 'Hi there!'."
)

result = agent.invoke({"input": task})
print(result)