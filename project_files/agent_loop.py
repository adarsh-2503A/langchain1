import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage,SystemMessage,ToolMessage
from langchain.tools import tool
from langsmith import traceable

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

MAX_ITERATIONS=10

if not api_key:
    raise ValueError("Missing Gemini API key. Add GEMINI_API_KEY to your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature = 0
)


@tool
def get_product_price(product:str)->float:
    """This tool provides the price of the product requested by the user"""
    print(f"Executing get_price tool for product-{product}")
    products_catalog={"headphones":12330,"smartphone":40000,"pendrive":3850.42}
    return products_catalog.get(product.strip().lower(), 0)

@tool
def apply_discount(price:float,tier:str)->float:
    """This tool applies the discount on the product for the respective tier"""
    print(f"Executing apply_discount tool for price-{price} and tier-{tier}")
    discounts={"silver":12,"gold":20,"platinum":24}
    return round(price*(1-discounts.get(tier,0)/100),2)

@traceable(name="Langchain agent loop")
def run_agent(question):
    tools=[get_product_price,apply_discount]
    tools_dict={t.name:t for t in tools}
    llm_with_tools=llm.bind_tools(tools=tools)

    print(f"Question: {question}")
    print("*"*75)
    messages=[SystemMessage(content=(
        """You are a helpful shopping assistant. You have access to a product catalog tool and a 
        discount tool.\n\nSTRICT RULES - you must follow these exactly:\n1. NEVER guess or assume any product price.\n
        2. You MUST call get_product_price first to get the real price.\n
        3. Only call apply_discount AFTER you have received a price from get_product_price. 
        Pass the exact price returned by get_product_price - do not pass a made-up number.\n
        3. NEVER calculate discounts yourself using math.
        \n4. Always use the apply_discount tool.\n4. If the user does not specify a discount tier,
        ask them which tier to use - do NOT assume one.""")),
        HumanMessage(content=question)
        ]
    for i in range(1,MAX_ITERATIONS+1):
        print(f"Iteration No-{i}")
        ai_message=llm_with_tools.invoke(messages)
        tool_calls=ai_message.tool_calls
        if not tool_calls:
            print(f"Final answer recieved: {ai_message.content}")
            return
        # Processing only one tool call- forcing one tool call per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        
        print(f"[Tool Selected] {tool_name} with args: {tool_args}")
        
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        observation = tool_to_use.invoke(tool_args)
        
        print(f"[Tool Result] {observation}")
        
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
        
    print("ERROR: Max iterations reached without a final answer")

run_agent(input('Enter your product query\n'))