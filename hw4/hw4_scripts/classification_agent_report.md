# AG News Zero-Shot Classification Agent

## Implementation

`classification_agent.py` uses LangChain's `create_agent` with a `ChatOpenAI`
client pointed at the Nautilus OpenAI-compatible endpoint
(`https://ellm.nrp-nautilus.io/v1`) and the `gpt-oss` model. Set `NRP_API_KEY`
to the Nautilus API key before running it. The script accepts one article as a
command-line argument or reads one article from standard input.

The agent uses zero-shot instructions: it receives the four labels and their
meanings, without labeled examples. It requests one brief evidence-based
justification and explicitly prohibits chain-of-thought.

## Prompt template

The template below is sent as the user's message to the agent, with `{article}`
replaced by the input text:

```text
Classify the AG News article below into exactly one category:
World, Sports, Business, or Sci/Tech.

Use these meanings:
- World: international or national politics, conflicts, society, and events.
- Sports: sports teams, athletes, competitions, and results.
- Business: companies, markets, finance, trade, and the economy.
- Sci/Tech: science, technology, research, and computing.

Return only a valid JSON object with exactly these two fields:
{"category": "one permitted category", "justification": "brief explanation"}
The category must exactly match one of the four names above. Keep the
justification to one concise sentence based on the article. Do not include
analysis, chain-of-thought, markdown, or any text outside the JSON object.

Article:
{article}
```

## Output validation

The response is parsed as JSON, then checked to ensure it is an object with
exactly `category` and `justification`. The category must exactly match one of
the four permitted labels. The justification must be a non-empty string of at
most 300 characters. Invalid output triggers one corrective retry. If that
response also fails validation, the script reports a validation error on
standard error and exits with status 2 instead of emitting invalid JSON.

Successful output is one JSON object, for example:

```json
{"category":"Sci/Tech","justification":"The article reports on a new computing technology."}
```
