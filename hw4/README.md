
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

| Model | Accuracy |
|---|---|
| TF-IDF MLP | 93.50% |
| gpt-oss agent | 86.00% |

The evaluation used 200 stratified AG News test examples with random seed 42.

Average agent latency: 0.90 seconds.

Total agent evaluation time: 180.91 seconds.

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
