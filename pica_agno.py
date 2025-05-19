import os
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.gmail import GmailTools
from langchain_openai import ChatOpenAI
from pica_langchain import PicaClient, create_pica_agent
from pica_langchain.models import PicaClientOptions
load_dotenv()
PICA_SECRET = os.getenv("PICA_SECRET")

class EmailData(BaseModel):
    message_id: str = Field(..., description="The message ID of the email")
    thread_id: str = Field(..., description="The thread ID of the email")
    references: Optional[str] = Field(None, description="The references of the email")
    in_reply_to: Optional[str] = Field(None, description="The in-reply-to of the email")
    subject: str = Field(..., description="The subject of the email")
    body: str = Field(..., description="The body of the email")
    sender: str = Field(..., description="The sender of the email")

class IntegratedEmailAgent:
    def __init__(self):
        
        self.agno_agent = Agent(
            name="Gmail Agent",
            model=OpenAIChat(id="gpt-4o"),
            tools=[GmailTools()],
            description="You are an expert Gmail Agent that can read, draft and send emails using Gmail.",
            instructions=[
                "Based on user query, you can read, draft and send emails using Gmail.",
                "While showing email contents, you can summarize the email contents, extract key details and dates.",
                "Show the email contents in a structured markdown format.",
            ],
            markdown=True,
            show_tool_calls=False,
            debug_mode=True,
            response_model=EmailData,
        )
        
        server_url = "https://development-api.picaos.com" 
        pica_options = PicaClientOptions(connectors=["*"])
        
        if server_url:
            pica_options.server_url = server_url
            
        self.pica_client = PicaClient(
            secret=PICA_SECRET,
            options=pica_options
        )
        
        llm = ChatOpenAI(temperature=0, model="gpt-4o")
        self.pica_agent = create_pica_agent(
            client=self.pica_client,
            llm=llm,
            verbose=True
        )
    
    def find_email(self, email_address: str) -> EmailData:
        response = self.agno_agent.run(
            f"Find the last email from {email_address} along with the message id, references and in-reply-to",
            markdown=True,
            stream=True,
        ).content
        
        return response
    
    def reply_to_email(self, email_data: EmailData, body: str) -> str:
        response = self.agno_agent.run(
            f"""Send an email in order to reply to the email.
            Use the thread_id {email_data.thread_id} and message_id {email_data.in_reply_to if email_data.in_reply_to else email_data.message_id}. 
            The subject should be 'Re: {email_data.subject}' and the body should be '{body}'""",
            markdown=True,
            stream=True,
        )
        
        return response
    
    def star_emails(self, email_address: str, count: int = 5) -> str:
        task = (
            f"Star emails from {email_address} and list {count} of the starred emails."
        )
        
        result = self.pica_agent.invoke({"input": task})
        return result["output"]
    
    def batch_operations(self, tasks: List[dict]) -> List[dict]:        
        results = []
        for task in tasks:
            result = self.pica_agent.invoke({"input": task["instruction"]})
            results.append({
                "task_id": task.get("id", "unknown"),
                "result": result["output"]
            })
        
        return results
    
    def execute_complex_workflow(self, workflow_instructions: str) -> dict:
        
        result = self.pica_agent.invoke({"input": workflow_instructions})
        
        relevant_info = self._extract_relevant_info(result["output"])
        
        if "email_address" in relevant_info:
            email_data = self.find_email(relevant_info["email_address"])
            if "reply_body" in relevant_info:
                reply_result = self.reply_to_email(email_data, relevant_info["reply_body"])
                return {
                    "pica_result": result["output"],
                    "agno_email_data": email_data.dict(),
                    "agno_reply_result": reply_result
                }
        
        return {"pica_result": result["output"]}
    
    def _extract_relevant_info(self, pica_output: str) -> dict:
        
        info = {}
        
        if "@" in pica_output:
            parts = pica_output.split()
            for part in parts:
                if "@" in part:
                    email = part.strip(",.!?()[]{}'\"\n")
                    if email.count("@") == 1 and "." in email.split("@")[1]:
                        info["email_address"] = email
                        break
        
        return info

if __name__ == "__main__":
    agent = IntegratedEmailAgent()
    
    email_address = "ameya@picaos.com"
    email_data = agent.find_email(email_address)
    print(f"Found email from {email_address}:")
    print(f"Subject: {email_data.subject}")
    print(f"Thread ID: {email_data.thread_id}")
    
    reply_result = agent.reply_to_email(email_data, "Hello, this is a test reply from the integrated agent.")
    print(f"Reply sent: {reply_result}")
    
    star_result = agent.star_emails("moe@picaos.com")
    print(f"Starred emails result: {star_result}")
    
    workflow = (
        "Find all emails from moe@picaos.com in the last week, "
        "star the important ones, and send a summary report to ameya@picaos.com"
    )
    workflow_result = agent.execute_complex_workflow(workflow)
    print(f"Workflow result: {workflow_result}")