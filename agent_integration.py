import os
from dotenv import load_dotenv

from agno.agent import Agent as AgnoAgent
from agno.models.openai import OpenAIChat
from agno.storage.sqlite import SqliteStorage
from agno.tools import tool
from agno.tools.reasoning import ReasoningTools

from pica_langchain import PicaClient, create_pica_agent
from pica_langchain.models import PicaClientOptions
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType

load_dotenv()
PICA_SECRET = os.getenv("PICA_SECRET")
if not PICA_SECRET:
    raise ValueError("PICA_SECRET environment variable is not set")

PICA_SERVER_URL = os.getenv("PICA_SERVER_URL", "https://development-api.picaos.com")

pica_client = PicaClient(
    secret=PICA_SECRET,
    options=PicaClientOptions(
        authkit=True,
        connectors=["*"],
        server_url=PICA_SERVER_URL
    )
)

pica_client.initialize()

llm = ChatOpenAI(temperature=0, model="gpt-4o")

pica_agent = create_pica_agent(
    client=pica_client,
    llm=llm,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=False
)

@tool
def use_pica_agent(task: str) -> str:
    """Use the PICA agent to perform tasks on connected services like Gmail, GitHub, etc.
    
    Args:
        task (str): The task to perform, e.g., "List 5 recent emails from Gmail"
    
    Returns:
        str: The result of the task
    """
    try:
        result = pica_agent.invoke({"input": task})
        return str(result)
    except Exception as e:
        return f"Error using PICA agent: {str(e)}"


def create_integrated_agent() -> AgnoAgent:
    """
    Create an Agno agent with reasoning, memory, and storage capabilities
    that can interact with PICA for connection-related tasks.
    
    Returns:
        An Agno agent configured with PICA integration
    """
    agno_agent = AgnoAgent(
        name="Agno-Pica Assistant",
        model=OpenAIChat(id="gpt-4o"),
        reasoning=True,  
        tools=[
            ReasoningTools(add_instructions=True),
            use_pica_agent,
        ],
        instructions=[
            "You are an intelligent assistant combining Agno's reasoning capabilities with PICA's connection handling.",
            "For tasks involving external services (e.g., Gmail, GitHub), use the 'use_pica_agent' tool to delegate to PICA.",
            "Use 'list_pica_connections' to check available connections if needed.",
            "If a required connection is missing, inform the user to set it up.",
            "Think step-by-step before responding, especially for complex tasks.",
            "Maintain conversation context using your memory capabilities.",
            "Be helpful, concise, and accurate in your responses.",
        ],
        storage=SqliteStorage(
            table_name="agno_pica_agent",
            db_file="agno_pica_agent.db",
            auto_upgrade_schema=True
        ),
        add_history_to_messages=True,
        num_history_responses=5,
        add_datetime_to_instructions=True,
        markdown=True,
        show_tool_calls=False,
        debug_mode=False,
    )
    
    return agno_agent

def main():
    """Main function to run the Agno-PICA integrated agent."""
    print("\n=== Agno-PICA Integration Terminal ===")
    print("(Type 'exit' to quit)")
    print("-------------------------------------")

    try:
        agent = create_integrated_agent()

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ['exit', 'quit']:
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue

                print("\nAgno: ", end="", flush=True)
                
                response = agent.run(
                    user_input,
                    stream=True,
                    show_full_reasoning=False,
                    verbose=False,
                    show_reasoning=False 
                )

                if hasattr(response, '__iter__'):
                    for chunk in response:
                        if chunk.content:
                            print(chunk.content, end="", flush=True)
                    print("\n")
                else:
                    print(response.content)
                    print("\n")

            except KeyboardInterrupt:
                print("\n\nExiting gracefully...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again.")

    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("Please check your environment variables and try again.")

if __name__ == "__main__":
    main()