from agno.agent import Agent
from agno.models.openai import OpenAIChat
from pica_langchain import create_pica_tools, PicaClient
from pica_langchain.models import PicaClientOptions

from dotenv import load_dotenv
import os

load_dotenv()
development_server_url = "https://development-api.picaos.com"
PICA_SECRET= os.getenv("PICA_SECRET")
pica_client = PicaClient(
    secret=PICA_SECRET,
    options=PicaClientOptions(connectors=["gmail"], server_url=development_server_url)
)

def wrap_tool(tool):
    def tool_func(*args, **kwargs):
        try:
            return tool.run(*args, **kwargs)
        except Exception:
            return tool.run(args[0] if args else kwargs)
    tool_func.__name__ = tool.name
    tool_func.description = getattr(tool, "description", tool.name)
    return tool_func

pica_tools = [wrap_tool(tool) for tool in create_pica_tools(pica_client)]

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=pica_tools,
    instructions=["Use the provided tools to manage Gmail tasks, such as starring emails or listing starred emails."],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

task = "Star emails from moe@picaos.com and list 5 of the starred emails. Then, send a reply to the starred emails with the subject 'Hello' and the body 'Hi there!'"
response = agent.print_response(task, stream=True)
print(response)