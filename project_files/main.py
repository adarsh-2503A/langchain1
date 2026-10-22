import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Missing Gemini API key. Add GEMINI_API_KEY to your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature = 0
)

template="""
using the given information - {information} 
generate 
1) a short summary
2) two interesting facts about the person
3) Date of birth if present in the information
"""

information="""
Pichai Sundararajan (born June 10, 1972), better known as Sundar Pichai,[a] is an Indian-American business executive who has been the CEO of Google since 2015 and the CEO of its parent company Alphabet Inc. since 2019.[4][5][6]

Pichai began his career as a materials engineer. Following a short stint at the management consulting firm McKinsey & Co., Pichai joined Google in 2004,[7] where he led the product management and innovation efforts for a suite of Google's client software products, including Google Chrome and ChromeOS, as well as being largely responsible for Google Drive. In addition, he went on to oversee the development of other applications such as Gmail and Google Maps.

As of February 2026, his net worth is estimated at US$1.6 billion
"""

prompt_template=PromptTemplate(input_variables=["information"],template=template)
chain= prompt_template | llm | StrOutputParser()
response=chain.invoke(input={"information":information})
print(response)