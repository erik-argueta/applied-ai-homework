# pip install google-genai
# How to get API key: Google AI studio (aistudio.google.com) or Google Cloud 

import os
from dotenv import load_dotenv
import json
from typing import Dict, Any

from google import genai
from google.genai import types

load_dotenv()

# 1. Initialize Google GenAI Client
client = genai.Client() # will read GOOGLE_API_KEY from environment variable

# ==========================================
# 2. Define Tools (Function Declarations)
# ==========================================
def search_market_trends(topic: str) -> str:
    """Retrieves market trends, growth metrics, and adoption statistics for a given topic."""
    return f"Market Data for '{topic}': CAGR estimated at 24.5%. Major adoption in FinTech and Healthcare. Key risk: Regulatory compliance."

def analyze_tech_feasibility(topic: str) -> str:
    """Analyzes technical feasibility, system requirements, and architecture constraints for a topic."""
    return f"Technical Feasibility for '{topic}': Scalable using Kubernetes + Vector DBs. Latency ~120ms. Bottleneck: GPU memory constraints."

# ==========================================
# 3. Step 1: Parallel Research Phase
# ==========================================
def run_parallel_research(topic: str, feedback: str = "") -> Dict[str, str]:
    """Runs Market and Technical research tasks concurrently using Gemini with Function Calling."""
    
    extra_context = f"\nAddress this feedback from previous review: {feedback}" if feedback else ""
    
    # --- Market Research Call ---
    market_prompt = (
        f"Analyze market trends for: '{topic}'.{extra_context}\n"
        "Use the search_market_trends tool to fetch data before providing your analysis."
    )
    market_response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=market_prompt,
        config=types.GenerateContentConfig(
            tools=[search_market_trends],
            system_instruction="You are a Market Research Analyst. Provide concise, data-driven insights."
        )
    )

    # --- Technical Audit Call ---
    tech_prompt = (
        f"Evaluate technical feasibility for: '{topic}'.{extra_context}\n"
        "Use the analyze_tech_feasibility tool to fetch data before providing your analysis."
    )
    tech_response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=tech_prompt,
        config=types.GenerateContentConfig(
            tools=[analyze_tech_feasibility],
            system_instruction="You are a Chief Technology Officer. Provide concise technical feasibility insights."
        )
    )

    return {
        "market_report": market_response.text,
        "tech_report": tech_response.text
    }

# ==========================================
# 4. Step 2: Synthesis Phase
# ==========================================
def run_synthesis(topic: str, market_report: str, tech_report: str) -> str:
    """Synthesizes research into a structured Executive Proposal."""
    prompt = (
        f"Synthesize the following research reports into an Executive Proposal for '{topic}'.\n\n"
        f"--- MARKET REPORT ---\n{market_report}\n\n"
        f"--- TECHNICAL REPORT ---\n{tech_report}\n\n"
        "Draft a structured summary with 3 sections: Executive Summary, Strategic Risks, and Next Steps."
    )
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are an Executive Business Strategist."
        )
    )
    return response.text

# ==========================================
# 5. Step 3: Evaluator / Quality Gate Phase
# ==========================================
def run_quality_gate(draft: str) -> Dict[str, Any]:
    """Evaluates the executive draft against compliance rules and returns structured output."""
    eval_prompt = (
        "Evaluate the proposal draft below based on these rules:\n"
        "1. Must explicitly mention 'Risks'.\n"
        "2. Must provide concrete next steps.\n\n"
        f"Draft:\n{draft}\n\n"
        "Respond in JSON format with two keys:\n"
        "- 'approved': boolean (true if passed, false otherwise)\n"
        "- 'feedback': string (empty if approved, or specific reasons for rejection)"
    )

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=eval_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )
    
    return json.loads(response.text)

# ==========================================
# 6. Workflow Loop Execution
# ==========================================
def execute_complex_agent_workflow(topic: str, max_revisions: int = 3):
    print(f"=== Starting Google ADK Workflow for Topic: '{topic}' ===\n")
    
    feedback = ""
    for revision in range(1, max_revisions + 1):
        print(f"--- [Attempt {revision}/{max_revisions}] Executing Workflow ---")
        
        # 1. Parallel Research
        print("  🔄 Executing Market & Technical Research...")
        reports = run_parallel_research(topic, feedback)
        
        # 2. Sequential Synthesis
        print("  📝 Synthesizing Executive Proposal...")
        draft = run_synthesis(topic, reports["market_report"], reports["tech_report"])
        
        # 3. Quality Gate Evaluation
        print("  🔍 Running Quality Gate Evaluation...")
        eval_result = run_quality_gate(draft)
        
        if eval_result.get("approved"):
            print("\n✅ [QUALITY GATE PASSED] Proposal approved!")
            print("\n=== FINAL EXECUTIVE PROPOSAL ===")
            print(draft)
            return draft
        else:
            feedback = eval_result.get("feedback", "Draft needs further refinement.")
            print(f"  ⚠️ [QUALITY GATE REJECTED] Feedback: {feedback}\n")
    
    print("⚠️ Reached max revision limits. Returning latest draft.")
    print(draft)
    return draft

# Execute the workflow
if __name__ == "__main__":
    topic_query = "Enterprise Deployment of Autonomous AI Agents in Healthcare"
    execute_complex_agent_workflow(topic_query)