import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel,Field

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str=Field(description="The URL of the source")
class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str=Field(description="The agent's answer to the query")
    sources:List[Source]=Field(default_factory=list,description="List of sources used to generate the answer")

load_dotenv()
# tavily=TavilyClient()
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature = 0
)

# @tool
# def search(query:str)->str:
#     """
#     This tool searches over the internet
#     Args: 
#         Query: the query to search for
#     Returns:
#         the search result
#     """
#     return tavily.search(query=query)

# agent=create_agent(model=llm,tools=[search])
agent=create_agent(model=llm,tools=[TavilySearch()],response_format=AgentResponse)
result=agent.invoke({"messages":HumanMessage(content="search for 3 recent job postings for an AI engineer in hyderabad and list their details")})
print(result)