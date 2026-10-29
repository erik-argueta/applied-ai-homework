# pip install crewai
# How to get API key: API keys for your LLMs

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool 

load_dotenv()

# ==========================================
# 1. Setup LLM configured for custom endpoint
# creai has its own LLM wrapper that can be used to connect to any LLM endpoint
# ==========================================
llm = LLM(
    model="kimi",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://ellm.nrp-nautilus.io/v1"
)

# ==========================================
# 2. Define Custom Tools
# ==========================================
@tool("fetch_sales_data")
def fetch_sales_data(quarter: str) -> str:
    """Retrieves financial sales data for a given quarter (e.g., 'Q1', 'Q2')."""
    data = {
        "Q1": "Revenue: $1.2M, Growth: +15%, Top Seller: Enterprise Tier",
        "Q2": "Revenue: $1.5M, Growth: +25%, Top Seller: AI Add-on Modules",
    }
    return data.get(quarter.upper(), f"No data found for quarter {quarter}.")

# ==========================================
# 3. Define Role-Based Agents
# ==========================================

# Researcher Agent
researcher = Agent(
    role="Financial Data Analyst",
    goal="Extract and verify raw sales metrics accurately using tools.",
    backstory=(
        "You are a meticulous financial analyst known for digging into quarterly "
        "reports and pulling clean, accurate data."
    ),
    tools=[fetch_sales_data],
    llm=llm,
    verbose=True
)

# Writer Agent
writer = Agent(
    role="Executive Communications Specialist",
    goal="Synthesize complex financial data into concise executive summaries.",
    backstory=(
        "You craft compelling, high-level C-suite summaries. You focus on key growth "
        "drivers, risks, and bullet points ready for email delivery."
    ),
    llm=llm,
    verbose=True
)

# ==========================================
# 4. Define Tasks & Assign Dependencies
# ==========================================

research_task = Task(
    description="Fetch the financial sales data for Q2.",
    expected_output="A summary of Q2 revenue, growth rate, and top selling products.",
    agent=researcher
)

writing_task = Task(
    description=(
        "Using the research gathered, write a polished 3-bullet point executive email summary. "
        "Keep the tone professional and focus on key wins."
    ),
    expected_output="A concise 3-bullet executive email summary ready for C-level leadership.",
    agent=writer
)

# ==========================================
# 5. Assemble and Run the Crew
# ==========================================

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential  # Runs tasks sequentially (Research -> Write)
)

print("=== Starting CrewAI Execution ===\n")
result = crew.kickoff()

print("\n=== FINAL RESULT ===")
print(result)