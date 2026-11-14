import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.neural_network import MLPClassifier

# these are our labels
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
MIN_DF = 2


# 1a loading dataset
def load_ag_news():
    """Returns the training_df and the testing_df with labeled columns."""
    from torchtext.datasets import AG_NEWS

    # combining each articles title  description
    training_rows = list(AG_NEWS(split="train"))
    testing_rows = list(AG_NEWS(split="test"))
    print("Loaded AG_NEWS via torchtext")
    mk = lambda rows: pd.DataFrame(
        {"label": [int(l) - 1 for l, _ in rows], "text": [t for _, t in rows]}
    )
    return mk(training_rows), mk(testing_rows)


def main():
    pd.set_option("display.width", 200)
    train_df, test_df = load_ag_news()
    print(f"\nTrain size: {len(train_df)} | Test size: {len(test_df)}")

    # displaying one example article from each category and the number of examples per category in both splits
    # 1c
    print("\n=== One example article per category (training set) ===")
    for k, name in enumerate(CLASS_NAMES):
        examples = train_df[train_df.label == k].iloc[0]["text"]
        print(f"[{k + 1}: {name}] {examples[:300]}")

    counts = pd.DataFrame(
        {
            "train": train_df.label.value_counts().sort_index().values,
            "test": test_df.label.value_counts().sort_index().values,
        },
        index=[f"{k + 1}: {c}" for k, c in enumerate(CLASS_NAMES)],
    )
    print("\n=== Examples per category ===")
    print(counts)

    train_text, test_text = train_df.text.tolist(), test_df.text.tolist()

    cv = CountVectorizer()
    training_cv = cv.fit_transform(train_text)
    testing_cv = cv.transform(test_text)

    tf = TfidfVectorizer()
    training_tf = tf.fit_transform(train_text)
    testing_tf = tf.transform(test_text)

    n_docs, n_feats = training_cv.shape
    sparsity = (1 - training_cv.nnz / (n_docs * n_feats)) * 100
    print(f"\n=== Training DTM (CountVectorizer, min_df={MIN_DF}) ===")
    print(f"Shape (documents x features): {n_docs} x {n_feats}")
    print(f"Non-zero entries: {training_cv.nnz:,} of {n_docs * n_feats:,} cells")
    print(f"Sparsity: {sparsity:.4f}%")
    print(f"Test DTM shape:     {testing_cv.shape}")
    print(f"TF-IDF train shape: {training_tf.shape} | test: {testing_tf.shape}")

    # preparing labels and model settings

    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    labels = list(range(len(CLASS_NAMES)))

    mlp_question1 = MLPClassifier(
        hidden_layer_sizes=(32,),
        max_iter=10,
        activation="relu",
        solver="adam",
        random_state=42,
        learning_rate_init=0.001,
        batch_size=1024,
        n_iter_no_change=10,
    )

    mlp_question1.fit(training_cv.astype(np.float32), y_train)
    y_pred = mlp_question1.predict(testing_cv.astype(np.float32))

    accuracy = accuracy_score(y_test, y_pred)
    macro_precision = precision_score(
        y_test, y_pred, labels=labels, average="macro", zero_division=0
    )

    print("\n--- CountVectorizer MLP Results ---")
    print(f"Architecture: {training_cv.shape[1]} -> 32 -> {len(CLASS_NAMES)}")
    print(f"Epochs completed: {mlp_question1.n_iter_}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro-precision: {macro_precision:.2%}")

    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion matrix (rows = true, columns = predicted)")
    print(pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES))

    print("\n--- Three Misclassified Articles ---")
    for i in np.flatnonzero(y_test != y_pred)[:3]:
        print(f"\nTrue category: {CLASS_NAMES[y_test[i]]}")
        print(f"Predicted category: {CLASS_NAMES[y_pred[i]]}")
        print(test_text[i])

    results = [
        {
            "Vectorizer": "CountVectorizer",
            "Stop-word removal": "No",
            "Accuracy": accuracy,
            "Macro-precision": macro_precision,
        }
    ]

    min_df = cv.min_df

    comparisons = [
        (TfidfVectorizer(min_df=min_df), "No"),
        (CountVectorizer(min_df=min_df, stop_words="english"), "Yes"),
        (TfidfVectorizer(min_df=min_df, stop_words="english"), "Yes"),
    ]

    for vectorizer, stop_word_removal in comparisons:
        name = type(vectorizer).__name__
        print(f"\nTraining {name} (stop-word removal: {stop_word_removal})", flush=True)

        # Learn the vocabulary and any IDF weights from training text.
        x_train = vectorizer.fit_transform(train_text).astype(np.float32)
        x_test = vectorizer.transform(test_text).astype(np.float32)

        # Train a fresh model with the baseline's settings.
        mlp = clone(mlp_question1)
        mlp.fit(x_train, y_train)
        predictions = mlp.predict(x_test)

        results.append(
            {
                "Vectorizer": name,
                "Stop-word removal": stop_word_removal,
                "Accuracy": accuracy_score(y_test, predictions),
                "Macro-precision": precision_score(
                    y_test,
                    predictions,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ),
            }
        )

    print("\n--- Feature Comparison ---")
    print(pd.DataFrame(results).to_string(index=False, float_format="{:.2%}".format))


if __name__ == "__main__":
    main()
