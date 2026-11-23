
import os
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

# Connect to the Nautilus gpt-oss model
llm = ChatOpenAI(
    model="gpt-oss",
    base_url="https://ellm.nrp-nautilus.io/v1",
    api_key=os.getenv("NRP_API_KEY"),
    temperature=0,
)

# Zero-shot classification prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a news classification agent.

Classify the given AG News article into exactly one category:
World, Sports, Business, or Sci/Tech.

Return ONLY a valid JSON object with exactly two fields:
"category" and "justification".

The category must be one of the four allowed categories.
The justification should briefly explain the classification.
Do not include markdown or extra text.
"""
    ),
    ("human", "Article:\n{article}")
])

# Create the LangChain classification pipeline
agent = prompt | llm


def classify_article(article):
    """Classify one news article and validate its response."""

    for attempt in range(2):
        try:
            response = agent.invoke({"article": article})
            result = json.loads(response.content)

            if not isinstance(result, dict):
                raise ValueError("Response must be a JSON object.")

            if set(result.keys()) != {"category", "justification"}:
                raise ValueError("Invalid JSON fields.")

            if result["category"] not in CATEGORIES:
                raise ValueError("Invalid category.")

            if not isinstance(result["justification"], str):
                raise ValueError("Justification must be text.")

            if not result["justification"].strip():
                raise ValueError("Justification cannot be empty.")

            return result

        except (json.JSONDecodeError, ValueError, TypeError) as error:
            if attempt == 1:
                return {
                    "error": f"Classification validation failed: {error}"
                }

    return {"error": "Classification failed."}


if __name__ == "__main__":

    article = (
        "NASA announced a new mission to explore Mars "
        "using advanced robotic technology."
    )

    print("Article:")
    print(article)

    print("\nClassification Result:")

    result = classify_article(article)

    print(json.dumps(result, indent=4))
