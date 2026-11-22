"""Zero-shot AG News classification with a LangChain agent on Nautilus.

The agent's final response must be a JSON object with exactly ``category`` and
``justification``.  If the first response is not valid, it is asked once to
correct its response; a second invalid response raises a clear validation
error rather than returning malformed data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


CATEGORIES = ("World", "Sports", "Business", "Sci/Tech")
NAUTILUS_BASE_URL = "https://ellm.nrp-nautilus.io/v1"

# Zero-shot prompt template. This is also reproduced in
# classification_agent_report.md for the assignment report.
CLASSIFICATION_PROMPT = """Classify the AG News article below into exactly one category:
World, Sports, Business, or Sci/Tech.

Use these meanings:
- World: international or national politics, conflicts, society, and events.
- Sports: sports teams, athletes, competitions, and results.
- Business: companies, markets, finance, trade, and the economy.
- Sci/Tech: science, technology, research, and computing.

Return only a valid JSON object with exactly these two fields:
{{"category": "one permitted category", "justification": "brief explanation"}}
The category must exactly match one of the four names above. Keep the
justification to one concise sentence based on the article. Do not include
analysis, chain-of-thought, markdown, or any text outside the JSON object.

Article:
{article}"""

REPAIR_PROMPT = """Your previous response did not satisfy the output contract.
Return only one valid JSON object with exactly the fields "category" and
"justification". The category must be exactly one of: World, Sports, Business,
Sci/Tech. The justification must be one brief sentence; do not include
analysis, chain-of-thought, markdown, or text outside the JSON.

Validation issue: {validation_error}
Previous response: {previous_response}
Article:
{article}"""


class ClassificationValidationError(ValueError):
    """Raised when the model fails the JSON contract after one retry."""


def _message_text(content: Any) -> str:
    """Extract plain text from LangChain's string or block-list content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "".join(text_parts)
    return str(content)


def _validate_response(raw_response: str) -> dict[str, str]:
    """Parse model output and enforce the exact response schema."""
    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ClassificationValidationError("response is not valid JSON") from exc

    if not isinstance(result, dict):
        raise ClassificationValidationError("response must be a JSON object")
    if set(result) != {"category", "justification"}:
        raise ClassificationValidationError(
            'JSON must contain exactly "category" and "justification"'
        )
    if not isinstance(result["category"], str) or result["category"] not in CATEGORIES:
        raise ClassificationValidationError(
            f"category must be one of: {', '.join(CATEGORIES)}"
        )
    justification = result["justification"]
    if not isinstance(justification, str) or not justification.strip():
        raise ClassificationValidationError("justification must be a non-empty string")
    if len(justification.strip()) > 300:
        raise ClassificationValidationError("justification must be brief (300 characters max)")

    return {"category": result["category"], "justification": justification.strip()}


def _build_agent():
    """Create a LangChain agent backed by the Nautilus gpt-oss endpoint."""
    api_key = os.environ.get("NRP_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set NRP_API_KEY (or OPENAI_API_KEY) to your Nautilus API key."
        )

    llm = ChatOpenAI(
        model="gpt-oss",
        temperature=0,
        api_key=api_key,
        base_url=os.environ.get("NAUTILUS_BASE_URL", NAUTILUS_BASE_URL),
    )
    # Classification needs no external tools; create_agent still provides the
    # LangChain agent execution and message handling requested by the assignment.
    return create_agent(llm, tools=[])


def classify_article(article: str, agent: Any | None = None) -> dict[str, str]:
    """Classify one article and return exactly ``category`` and ``justification``.

    ``agent`` may be supplied by callers that want to reuse a configured agent.
    Invalid model output is retried once, then reported as a clear exception.
    """
    if not isinstance(article, str) or not article.strip():
        raise ValueError("article must be a non-empty string")

    classifier = agent if agent is not None else _build_agent()
    prompt = CLASSIFICATION_PROMPT.format(article=article.strip())
    previous_response = ""
    validation_error = ""

    for attempt in range(2):
        current_prompt = prompt if attempt == 0 else REPAIR_PROMPT.format(
            validation_error=validation_error,
            previous_response=previous_response,
            article=article.strip(),
        )
        response = classifier.invoke({"messages": [("user", current_prompt)]})
        previous_response = _message_text(response["messages"][-1].content)
        try:
            return _validate_response(previous_response)
        except ClassificationValidationError as exc:
            validation_error = str(exc)

    raise ClassificationValidationError(
        "Model returned invalid classification JSON after one retry: "
        f"{validation_error}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify one AG News article with Nautilus gpt-oss."
    )
    parser.add_argument(
        "article",
        nargs="?",
        help="the article text; if omitted, read the article from standard input",
    )
    args = parser.parse_args()
    article = args.article if args.article is not None else sys.stdin.read()

    try:
        result = classify_article(article)
    except ClassificationValidationError as exc:
        print(f"Classification validation error: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as exc:
        print(f"Classification error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
