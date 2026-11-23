
# Assignment 4 — Parts 2 and 3

## Overview

This project implements an LLM-based news classification agent and a Retrieval-Augmented Generation (RAG) application.

Part 2 classifies AG News articles using the gpt-oss model on Nautilus and compares its performance against a traditional MLP classifier.

Part 3 uses the CPSC 254 course syllabus as a knowledge source to answer questions through a ChromaDB vector database and the gpt-oss model.

## Dependencies

Install the required packages:

```bash
python -m pip install langchain-openai langchain-core python-dotenv scikit-learn pandas numpy datasets chromadb pymupdf tiktoken openai
```

## Part 2: Classification Agent

Run the classification agent:

```bash
python hw4/hw4_scripts/classification_agent.py
```

Run the 200-article evaluation:

```bash
python hw4/hw4_scripts/classification_eval.py
```

Results are saved to `hw4/hw4_scripts/part2_results/`.

### Part 2 Results

The comparison below uses the same 200 stratified examples from the official
AG News test split (random seed 42; 50 examples per category). It reports the
saved evaluation in `hw4/hw4_scripts/part2_results/`; the 200 test examples
were used only for evaluation, not tuning. The traditional classifier is the
Part 1 TF-IDF MLP, trained on the 120,000 AG News training examples.

| Method | Accuracy on same 200 examples | Agent latency / evaluation time | API cost / compute | Data efficiency | Explanations and error analysis |
|---|---:|---|---|---|---|
| Zero-shot `gpt-oss` agent | 86.00% (172/200); 0 invalid outputs | Mean: 0.904 s/example; total agent loop: 180.91 s | Cost unavailable: the Nautilus-compatible endpoint and saved run do not expose billing/token pricing. Model inference ran on Nautilus; local GPU requirements are not applicable. | Zero AG News labeled examples in the prompt; uses the pretrained model's prior training. | Brief justifications make individual predictions easier to inspect, but can be generic or incorrect and are not a calibrated explanation. |
| TF-IDF MLP (Part 1) | 93.50% (187/200) | Not measured separately; evaluation script times the agent loop, not MLP fit or prediction. | No API cost. Scikit-learn model runs locally on CPU; actual CPU/RAM configuration was not recorded. | Supervised training on all 120,000 labeled AG News training examples. | No natural-language justification. Confusion matrix and misclassified examples expose error patterns, but do not explain an individual decision. |

The MLP was 7.5 percentage points more accurate on this sample. Its 120,000
training examples are substantially more labeled task data than the agent's
zero-shot prompt uses. The MLP's training time was not included in the agent's
180.91-second evaluation time. Cost was not available from the evaluation
artifacts, which record latency and predictions but no API billing data.

The evaluation script configures `TfidfVectorizer(min_df=2)`. In the Part 1
comparison script, the best representation was TF-IDF, but its vectorizer
inherits `min_df=1` from the default `CountVectorizer()` because the declared
`MIN_DF=2` is not passed to that vectorizer. Thus the saved comparison uses
the same Part 1 classifier family and selected representation, with this
small vocabulary-threshold difference.

## Part 3: RAG Application

The RAG application uses the CPSC 254 syllabus PDF as its knowledge source.

Run from the RAG directory:

```bash
cd hw4/hw4_scripts/rag
python rag.py syllabus.pdf
```

The application extracts syllabus text, splits it into overlapping chunks, stores embeddings in ChromaDB, and retrieves relevant information to answer questions.

The implementation uses a target chunk size of 200 tokens and an overlap of 40 tokens.

### RAG Evaluation

From the repository root, run:

```bash
python hw4/hw4_scripts/rag_eval.py
```

The evaluation asks three syllabus questions about grading, attendance, and assignment requirements.

Results are saved to `hw4/hw4_scripts/rag_evaluation_results.csv`.

### RAG Evaluation Ratings

| Question | Accuracy | Completeness | Relevance |
|---|---:|---:|---:|
| Grading policy | 2 | 2 | 2 |
| Attendance policy | 1 | 1 | 2 |
| Assignment requirements | 2 | 2 | 2 |

Ratings are preliminary and should be verified against the original syllabus.
