
import csv
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from classification_agent import classify_article, CATEGORIES


OUTPUT_DIR = Path(__file__).resolve().parent / "part2_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("Loading AG News dataset...", flush=True)

    dataset = load_dataset("fancyzhx/ag_news")

    train_text = dataset["train"]["text"]
    train_labels = np.array(dataset["train"]["label"])

    test_text = dataset["test"]["text"]
    test_labels = np.array(dataset["test"]["label"])

    # Stratified sample: 50 examples per category
    indices = np.arange(len(test_labels))

    _, sample_indices = train_test_split(
        indices,
        test_size=200,
        stratify=test_labels,
        random_state=42,
    )

    sample_text = [test_text[i] for i in sample_indices]
    sample_labels = test_labels[sample_indices]

    print("Sample size:", len(sample_indices))
    print("Training TF-IDF MLP...", flush=True)

    vectorizer = TfidfVectorizer(min_df=2)

    X_train = vectorizer.fit_transform(train_text)
    X_sample = vectorizer.transform(sample_text)

    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        max_iter=10,
        activation="relu",
        solver="adam",
        random_state=42,
        learning_rate_init=0.001,
        batch_size=1024,
        n_iter_no_change=10,
    )

    model.fit(X_train.astype(np.float32), train_labels)

    mlp_predictions = model.predict(
        X_sample.astype(np.float32)
    )

    mlp_accuracy = accuracy_score(
        sample_labels, mlp_predictions
    )

    mlp_precision = precision_score(
        sample_labels,
        mlp_predictions,
        average="macro",
        zero_division=0,
    )

    print(f"MLP accuracy: {mlp_accuracy:.2%}")
    print(f"MLP macro-precision: {mlp_precision:.2%}")

    print("\nEvaluating gpt-oss agent...", flush=True)

    agent_predictions = []
    records = []
    latencies = []

    start_total = time.perf_counter()

    for i, (article, true_label) in enumerate(
        zip(sample_text, sample_labels), start=1
    ):
        start = time.perf_counter()

        try:
            result = classify_article(article)
        except Exception as error:
            result = {"error": str(error)}

        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        predicted_category = result.get("category")

        if predicted_category in CATEGORIES:
            predicted_label = CATEGORIES.index(
                predicted_category
            )
        else:
            predicted_label = -1

        agent_predictions.append(predicted_label)

        records.append({
            "article_number": i,
            "true_category": CATEGORIES[int(true_label)],
            "mlp_category": CATEGORIES[int(
                mlp_predictions[i - 1]
            )],
            "agent_category": (
                predicted_category
                if predicted_label != -1 else "INVALID"
            ),
            "justification": result.get(
                "justification", result.get("error", "")
            ),
            "latency_seconds": round(elapsed, 3),
        })

        print(
            f"{i}/200 | "
            f"True: {CATEGORIES[int(true_label)]} | "
            f"Agent: {records[-1]['agent_category']} | "
            f"Time: {elapsed:.2f}s",
            flush=True,
        )

    total_time = time.perf_counter() - start_total

    agent_accuracy = accuracy_score(
        sample_labels, agent_predictions
    )

    average_latency = np.mean(latencies)

    summary = {
        "sample_size": 200,
        "random_seed": 42,
        "mlp_accuracy": float(mlp_accuracy),
        "mlp_macro_precision": float(mlp_precision),
        "agent_accuracy": float(agent_accuracy),
        "agent_average_latency_seconds": float(
            average_latency
        ),
        "agent_total_time_seconds": float(total_time),
        "invalid_agent_responses": sum(
            p == -1 for p in agent_predictions
        ),
    }

    with open(
        OUTPUT_DIR / "evaluation_results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=records[0].keys()
        )
        writer.writeheader()
        writer.writerows(records)

    with open(
        OUTPUT_DIR / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n--- PART 2 COMPARISON ---")
    print(f"Sample size: 200")
    print(f"MLP accuracy: {mlp_accuracy:.2%}")
    print(f"Agent accuracy: {agent_accuracy:.2%}")
    print(f"Average agent latency: {average_latency:.2f}s")
    print(f"Total agent time: {total_time:.2f}s")
    print(
        f"Invalid responses: "
        f"{summary['invalid_agent_responses']}"
    )
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
