# Creating a simple agent
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

# 1. Define tools for the agent using the @tool decorator
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [multiply, get_word_length]

# 2. Initialize the LLM with tool-calling capabilities
llm = ChatOpenAI(
    model = "kimi",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://ellm.nrp-nautilus.io/v1"
)

# 3. Create an agent and workflow
agent = create_agent(llm, tools)

# 4. Run the agent with a prompt that requires tools
query = "What is the length of the word 'LangChain' multiplied by 5?"

response = agent.invoke({
    "messages": [("user", query)]
})

# 5. Print the agent's final answer
print(response["messages"][-1].content)
